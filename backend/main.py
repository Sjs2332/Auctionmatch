#!/usr/bin/env python3
"""
Copart Dealer Analysis and Watchlist Generator

This script analyzes dealership purchase patterns to create a scored watchlist
from Copart auction data for strategic bidding.
"""

import argparse
import json
import logging
import sys

import pandas as pd

from analysis_service import execute_analysis
from config import AnalyzerConfig
from utils import setup_logging, get_default_inventory_file


DEFAULT_INVENTORY_FILE = get_default_inventory_file()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Analyze dealer patterns and generate Copart watchlist"
    )
    parser.add_argument(
        "--dealer-files", 
        nargs="+", 
        default=[
            "LotsWon_sample.csv"
        ],
        help="Dealer purchase history CSV files"
    )
    parser.add_argument(
        "--inventory-file",
        default=DEFAULT_INVENTORY_FILE or "salesdata.csv",
        help="Copart inventory data CSV file (defaults to bundled salesdata.csv when available)"
    )
    parser.add_argument(
        "--output-file",
        default="sniper_watchlist.csv",
        help="Output watchlist CSV file"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=7.0,
        help="Minimum score for watchlist inclusion"
    )
    parser.add_argument(
        "--min-odometer",
        type=int,
        default=80000,
        help="Minimum odometer reading"
    )
    parser.add_argument(
        "--max-odometer", 
        type=int,
        default=150000,
        help="Maximum odometer reading"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Maximum number of vehicles to surface"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--config-file",
        help="Optional JSON file with analysis configuration overrides"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    analysis_config = AnalyzerConfig()
    if args.config_file:
        try:
            with open(args.config_file, "r", encoding="utf-8") as fh:
                overrides = json.load(fh)
            analysis_config.apply_overrides(overrides)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load config overrides: %s", exc)
            sys.exit(1)

    analysis_config.recall.base_min_score = max(analysis_config.recall.base_min_score, args.min_score)

    inventory_file = args.inventory_file or get_default_inventory_file()
    if not inventory_file:
        logger.error(
            "No inventory file provided and bundled default (salesdata.csv) was not found. "
            "Provide --inventory-file or set ROBONIX_DEFAULT_INVENTORY."
        )
        sys.exit(1)

    try:
        result = execute_analysis(
            dealer_files=args.dealer_files,
            inventory_file=inventory_file,
            min_score=args.min_score,
            min_odometer=args.min_odometer,
            max_odometer=args.max_odometer,
            top_n=args.top_n,
            log_level=args.log_level,
            config=analysis_config,
        )
    except ValueError as exc:
        logger.error("Analysis failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected failure during analysis")
        sys.exit(1)

    watchlist_data = result.get("watchlist", {})
    top_results = watchlist_data.get("top_results", [])
    watchlist_df = pd.DataFrame(top_results)

    if watchlist_df.empty:
        logger.warning("No vehicles met the criteria after adaptive tuning.")
    else:
        watchlist_df.to_csv(args.output_file, index=False)
        logger.info("Exported %s vehicles to %s", len(watchlist_df), args.output_file)

    # Print concise summary
    print("\n" + "=" * 80)
    print("AUCTIONMATCH WATCHLIST SUMMARY")
    print("=" * 80)

    inventory_metrics = result.get("inventory", {})
    print(f"Filtered Vehicles: {inventory_metrics.get('filtered_count', 0):,}")
    if 'active_filter_settings' in inventory_metrics:
        settings = inventory_metrics['active_filter_settings']
        print(f"Adaptive Filter Level: {settings.get('filter_level', 'n/a')} (attempt {settings.get('attempt', 'n/a')})")

    print(f"Primary Picks: {len(watchlist_data.get('primary', []))}")
    print(f"Secondary Picks: {len(watchlist_data.get('secondary', []))}")
    score_stats = watchlist_data.get('score_stats') or {}
    if score_stats:
        print(f"Score Range: {score_stats.get('min', 0):.2f} – {score_stats.get('max', 0):.2f} (avg {score_stats.get('average', 0):.2f})")

    print("\nTop Primary Vehicles:")
    for entry in watchlist_data.get('primary', [])[:5]:
        print(
            f"  {entry.get('auction_date', 'Unknown')} @ {entry.get('auction_time', 'Unknown')} | "
            f"{entry.get('year_make_model', 'N/A')} | Location: {entry.get('location', 'Unknown')} | "
            f"Miles: {entry.get('miles', 'Unknown')} | Damage: {entry.get('damage', 'N/A')} | Cap: {entry.get('bid_cap', 'N/A')}"
        )
        if entry.get('lot_link'):
            print(f"    Lot: {entry['lot_link']}")
        if entry.get('image_link'):
            print(f"    Image: {entry['image_link']}")
        reasoning = entry.get('reasoning')
        if reasoning:
            print(f"    Reasoning: {reasoning}")

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
