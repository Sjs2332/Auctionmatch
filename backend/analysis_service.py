"""
Shared analysis workflow for API and CLI entrypoints.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from bidding_analyzer import BiddingAnalyzer
from dealer_analyzer import DealerAnalyzer
from inventory_processor import InventoryProcessor
from scoring_engine import ScoringEngine
from utils import validate_files
from config import AnalyzerConfig

logger = logging.getLogger(__name__)


def execute_analysis(
    dealer_files: List[str],
    inventory_file: str,
    min_score: float,
    min_odometer: int,
    max_odometer: int,
    top_n: int,
    log_level: str,
    config: AnalyzerConfig,
) -> Dict[str, Any]:
    """
    Perform the full end-to-end analysis and return structured results.
    """
    logging.getLogger().setLevel(getattr(logging, log_level))

    file_bundle = dealer_files + [inventory_file]
    if not validate_files(file_bundle):
        raise ValueError("One or more provided files are missing, unreadable, or invalid CSV.")

    analyzer = DealerAnalyzer()
    patterns = analyzer.analyze_patterns(dealer_files)
    if not patterns:
        raise ValueError("No dealer patterns could be derived from the supplied data.")

    inventory_processor = InventoryProcessor(patterns, config=config)
    filtered_inventory = inventory_processor.filter_inventory(
        inventory_file,
        min_odometer=min_odometer,
        max_odometer=max_odometer,
    )

    if filtered_inventory.empty:
        raise ValueError("No inventory vehicles matched the configured filters.")

    bidding_analyzer = BiddingAnalyzer(config=config)
    bidding_summary = bidding_analyzer.analyze_purchase_patterns(dealer_files)

    scoring_engine = ScoringEngine(patterns, bidding_analyzer, config=config)
    scored_inventory = scoring_engine.score_vehicles(filtered_inventory)
    sorted_inventory = scored_inventory.sort_values("score", ascending=False).reset_index(drop=True)

    recall_cfg = config.recall
    diversity_cfg = config.diversity

    effective_top_n = top_n or recall_cfg.max_results
    target_cap = min(recall_cfg.max_results, effective_top_n, diversity_cfg.secondary_count)
    if target_cap >= recall_cfg.target_results:
        target_total = recall_cfg.target_results
    else:
        target_total = target_cap

    if target_cap < recall_cfg.min_results:
        target_total = target_cap
    else:
        target_total = max(recall_cfg.min_results, target_total)

    requested_threshold = max(min_score, recall_cfg.base_min_score)
    current_threshold = requested_threshold

    def _eligible(threshold: float) -> pd.DataFrame:
        return sorted_inventory[sorted_inventory["score"] >= threshold]

    eligible = _eligible(current_threshold)
    adjusted_threshold = False

    while len(eligible) < recall_cfg.min_results and current_threshold > recall_cfg.min_score_floor:
        current_threshold = max(recall_cfg.min_score_floor, current_threshold - recall_cfg.score_step)
        new_eligible = _eligible(current_threshold)
        if len(new_eligible) == len(eligible):
            break
        eligible = new_eligible
        adjusted_threshold = True

    while len(eligible) > recall_cfg.max_results and current_threshold < requested_threshold + 10:
        current_threshold += recall_cfg.score_step
        new_eligible = _eligible(current_threshold)
        if len(new_eligible) == len(eligible):
            break
        eligible = new_eligible
        adjusted_threshold = True

    if config.logging.log_threshold_adjustments and adjusted_threshold:
        logger.info(
            "Adaptive threshold adjusted",
            extra={
                'requested_min_score': min_score,
                'final_threshold': current_threshold,
                'eligible_count': len(eligible)
            }
        )

    if eligible.empty:
        primary_df = pd.DataFrame()
        secondary_df = pd.DataFrame()
        combined_df = pd.DataFrame()
    else:
        cap_share = diversity_cfg.max_make_share
        make_cap = max(1, int(target_total * cap_share))

        selected_indices: List[int] = []
        overflow_indices: List[int] = []
        make_counts: Dict[str, int] = defaultdict(int)

        for idx, row in eligible.iterrows():
            if len(selected_indices) >= target_total:
                break
            make = str(row.get('Make', 'UNKNOWN')).upper()

            if diversity_cfg.max_make_share < 1.0:
                if make_counts[make] >= make_cap and len(selected_indices) >= diversity_cfg.enforce_after:
                    overflow_indices.append(idx)
                    continue

            selected_indices.append(idx)
            make_counts[make] += 1

        if diversity_cfg.max_make_share < 1.0 and len(selected_indices) < recall_cfg.min_results:
            for idx in overflow_indices:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) >= recall_cfg.min_results:
                        break

        selected_df = eligible.loc[selected_indices].sort_values("score", ascending=False)
        if len(selected_df) < recall_cfg.min_results:
            selected_df = eligible.head(min(target_total, len(eligible)))

        primary_limit = min(diversity_cfg.primary_count, len(selected_df))
        secondary_limit = max(0, min(target_total, len(eligible)) - primary_limit)

        primary_df = selected_df.head(primary_limit)
        secondary_df = selected_df.iloc[primary_limit:primary_limit + secondary_limit]

        if len(secondary_df) < secondary_limit:
            remaining = secondary_limit - len(secondary_df)
            extra_candidates = eligible[~eligible.index.isin(selected_df.index)].head(remaining)
            if not extra_candidates.empty:
                secondary_df = pd.concat([secondary_df, extra_candidates])

        combined_df = pd.concat([primary_df, secondary_df], ignore_index=True) if not primary_df.empty or not secondary_df.empty else pd.DataFrame()

    def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if df.empty:
            return []
        safe_df = df.where(pd.notnull(df), None)
        return safe_df.to_dict(orient="records")

    def _format_date(raw_date) -> str:
        if raw_date in (None, "", 0):
            return "Unknown"
        raw = str(raw_date).strip()
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
        return raw

    def _format_time(raw_time, timezone) -> str:
        if raw_time in (None, "", 0):
            return "Unknown"
        try:
            hhmm = f"{int(float(raw_time)):04d}"
            formatted = f"{hhmm[:2]}:{hhmm[2:]}"
        except (ValueError, TypeError):
            formatted = str(raw_time)
        if timezone:
            return f"{formatted} {timezone}"
        return formatted

    def _short_reason(score_breakdown: Optional[str], bid_reason: Optional[str]) -> str:
        parts = []
        if score_breakdown:
            parts.append(score_breakdown)
        if bid_reason:
            parts.append(bid_reason)
        summary = " | ".join(parts)
        return summary[:240] + ("…" if len(summary) > 240 else "") if summary else "Data-backed match"

    def _format_watchlist(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        placeholder_image = "https://cs.copart.com/v1/AUTH_svc.pdoc00001/ids-c-prod-lpp/default_placeholder.jpg"

        for rec in records:
            lot_number = rec.get('Lot number') or rec.get('Lot Number')
            lot_link = rec.get('copart_link') or (f"https://www.copart.com/lot/{int(float(lot_number))}" if lot_number not in (None, "") else "")
            timezone = rec.get('Time Zone') or rec.get('Time zone')
            damage = rec.get('Damage Description') or rec.get('Damage') or "UNKNOWN"
            score_val = rec.get('score')
            reasoning = _short_reason(rec.get('score_breakdown'), rec.get('bid_reasoning'))
            if score_val not in (None, ''):
                try:
                    reasoning = f"Score {float(score_val):.1f} | {reasoning}"
                except (ValueError, TypeError):
                    reasoning = f"Score {score_val} | {reasoning}"

            yard_name = rec.get('Yard name') or rec.get('Location') or rec.get('Location city')
            state = rec.get('Location state') or rec.get('Sale Title State')
            location_parts = []
            if yard_name:
                location_parts.append(str(yard_name).strip())
            if state:
                state_str = str(state).strip()
                if state_str and (not location_parts or state_str not in location_parts[0]):
                    location_parts.append(state_str)
            location = " | ".join(location_parts) if location_parts else "Unknown"

            odometer = rec.get('Odometer')
            if odometer in (None, "") or (isinstance(odometer, float) and np.isnan(odometer)):
                miles = "Unknown"
            else:
                try:
                    miles = f"{int(float(odometer)):,} mi"
                except (ValueError, TypeError):
                    miles = str(odometer)

            raw_thumbnail = rec.get('Image Thumbnail')
            raw_image_url = rec.get('Image URL')

            image_link = None
            if raw_thumbnail not in (None, ""):
                thumb_str = str(raw_thumbnail).strip()
                if thumb_str:
                    if not thumb_str.lower().startswith("http"):
                        thumb_str = f"https://{thumb_str.lstrip('/')}"
                    image_link = thumb_str
            elif raw_image_url not in (None, ""):
                url_str = str(raw_image_url).strip()
                if url_str:
                    if not url_str.lower().startswith("http"):
                        url_str = f"https://{url_str.lstrip('/')}"
                    image_link = url_str

            if not image_link:
                image_link = placeholder_image

            formatted.append({
                "auction_date": _format_date(rec.get('Sale Date M/D/CY') or rec.get('Sale Date')),
                "auction_time": _format_time(rec.get('Sale time (HHMM)') or rec.get('Sale Time'), timezone),
                "lot_link": lot_link,
                "lot_number": lot_number,
                "image_link": image_link,
                "year_make_model": " ".join(filter(None, [
                    str(rec.get('Year')) if rec.get('Year') not in (None, "") else None,
                    str(rec.get('Make')) if rec.get('Make') not in (None, "") else None,
                    str(rec.get('Model Group') or rec.get('Model')) if (rec.get('Model Group') or rec.get('Model')) else None,
                ])).strip() or "UNKNOWN",
                "damage": damage,
                "bid_cap": rec.get('bid_up_to', 'N/A'),
                "reasoning": reasoning,
                "location": location,
                "miles": miles,
            })
        return formatted

    primary_raw = _df_to_records(primary_df)
    secondary_raw = _df_to_records(secondary_df)
    primary_records = _format_watchlist(primary_raw)
    secondary_records_all = _format_watchlist(secondary_raw)
    primary_ids = {
        entry.get('lot_number') or entry.get('lot_link')
        for entry in primary_records
        if (entry.get('lot_number') or entry.get('lot_link'))
    }
    secondary_records = [
        entry for entry in secondary_records_all
        if (entry.get('lot_number') or entry.get('lot_link')) not in primary_ids
    ]
    seen_lots = set()
    combined_records = []
    for entry in primary_records + secondary_records:
        lot_key = entry.get('lot_number') or entry.get('lot_link')
        if lot_key and lot_key in seen_lots:
            continue
        if lot_key:
            seen_lots.add(lot_key)
        combined_records.append(entry)

    inventory_metrics = {
        "total_count": int(inventory_processor.total_records),
        "filtered_count": int(len(filtered_inventory)),
        "dynamic_filters": inventory_processor.dynamic_filters,
    }

    if not filtered_inventory.empty:
        if "Year" in filtered_inventory.columns:
            years = filtered_inventory["Year"].dropna()
            if not years.empty:
                inventory_metrics["year_span"] = {
                    "min": int(years.min()),
                    "max": int(years.max()),
                }
        if "Odometer" in filtered_inventory.columns:
            odometers = filtered_inventory["Odometer"].dropna()
            if not odometers.empty:
                inventory_metrics["odometer_span"] = {
                    "min": float(odometers.min()),
                    "max": float(odometers.max()),
                }

    active_filter_settings = inventory_processor.dynamic_filters.get('active_filter_settings')
    if active_filter_settings:
        inventory_metrics['active_filter_settings'] = active_filter_settings

    combined_count = len(combined_df) if not combined_df.empty else 0
    watchlist_summary: Dict[str, Any] = {
        "requested_min_score": min_score,
        "adaptive_min_score": current_threshold,
        "result_count": int(combined_count),
        "primary": primary_records,
        "secondary": [rec for rec in secondary_records if rec not in primary_records],
        "top_results": combined_records,
        "diversity_cap": config.diversity.max_make_share,
        "target_total": target_total,
    }
    watchlist_summary["primary_count"] = len(primary_records)
    watchlist_summary["secondary_count"] = len(watchlist_summary["secondary"])

    if not combined_df.empty:
        watchlist_summary["score_stats"] = {
            "min": float(combined_df["score"].min()),
            "max": float(combined_df["score"].max()),
            "average": float(combined_df["score"].mean()),
        }
        make_distribution = (
            combined_df['Make'].fillna('UNKNOWN')
            .str.upper()
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )
        watchlist_summary["make_distribution"] = make_distribution

    if config.logging.log_selection_reasons:
        for _, row in primary_df.iterrows():
            logger.info(
                "Primary recommendation",
                extra={
                    'lot': row.get('Lot number'),
                    'score': row.get('score'),
                    'make': row.get('Make'),
                    'model': row.get('Model Group'),
                    'reason': row.get('score_breakdown'),
                }
            )
        for _, row in secondary_df.iterrows():
            logger.info(
                "Secondary recommendation",
                extra={
                    'lot': row.get('Lot number'),
                    'score': row.get('score'),
                    'make': row.get('Make'),
                    'model': row.get('Model Group'),
                    'reason': row.get('score_breakdown'),
                }
            )

    return {
        "dealer_patterns": patterns,
        "inventory": inventory_metrics,
        "bidding_analysis": bidding_summary,
        "watchlist": watchlist_summary,
    }
