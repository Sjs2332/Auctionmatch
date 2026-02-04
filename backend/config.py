"""
Central configuration models for the AuctionMatch analysis pipeline.

All tunable knobs live here so workflows can be adjusted without code edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class RecallConfig:
    """Controls adaptive recall behaviour and score thresholds."""

    min_results: int = 30
    target_results: int = 50
    max_results: int = 60
    base_min_score: float = 7.0
    min_score_floor: float = 5.0
    score_step: float = 0.5
    odometer_step: int = 10000
    min_odometer_floor: int = 40000
    max_odometer_ceiling: int = 220000
    year_expansion_step: int = 1
    max_year_window: int = 10
    max_relax_attempts: int = 4


@dataclass
class DiversityConfig:
    """Keeps the watchlist balanced across makes and trims."""

    max_make_share: float = 1.0  # Allow full concentration by default
    primary_count: int = 15      # A-list target
    secondary_count: int = 50    # Total list size cap (includes primary)
    enforce_after: int = 50      # Only enforce caps very late (effectively off)


@dataclass
class SimilarityConfig:
    """Weights for similarity-driven scoring."""

    make_weight: float = 2.0
    model_weight: float = 1.2
    year_weight: float = 0.9
    location_weight: float = 0.6
    damage_weight: float = 0.7
    year_decay_per_step: float = 0.12  # Penalty per year away from ideal


@dataclass
class BidAdjustmentConfig:
    """Percentile bid caps with contextual modifiers."""

    mileage_penalty_threshold: int = 150000
    mileage_penalty_multiplier: float = 0.9
    low_mileage_bonus_threshold: int = 90000
    low_mileage_bonus_multiplier: float = 1.05
    non_preferred_location_penalty: float = 0.92
    preferred_location_bonus: float = 1.03
    distance_unknown_penalty: float = 0.95


@dataclass
class LoggingConfig:
    """Controls behavioural logging for explainability."""

    log_selection_reasons: bool = True
    log_threshold_adjustments: bool = True


@dataclass
class AnalyzerConfig:
    """Top-level configuration bundle."""

    recall: RecallConfig = field(default_factory=RecallConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    bid_adjustments: BidAdjustmentConfig = field(default_factory=BidAdjustmentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Return a nested dict representation."""
        return asdict(self)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply nested overrides to the config in-place."""

        def _apply(obj: Any, updates: Dict[str, Any]) -> None:
            for key, value in updates.items():
                if not hasattr(obj, key):
                    continue
                current = getattr(obj, key)
                if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
                    _apply(current, value)
                else:
                    setattr(obj, key, value)

        _apply(self, overrides or {})


DEFAULT_ANALYZER_CONFIG = AnalyzerConfig()
