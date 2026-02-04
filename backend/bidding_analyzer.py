"""
Bidding Analysis Module for Copart Dealer Analysis

This module analyzes historical dealer purchase data to calculate data-backed bidding limits
that maintain discipline and prevent overbidding based on successful dealer patterns.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from config import AnalyzerConfig

logger = logging.getLogger(__name__)


class BiddingAnalyzer:
    """Analyzes dealer purchase patterns to establish disciplined bidding limits"""
    
    def __init__(self, config: AnalyzerConfig = None):
        self.purchase_data = None
        self.bidding_rules = {}
        self.make_model_stats = {}
        self.year_stats = {}
        self.damage_stats = {}
        self.location_stats = {}
        self.location_preferences = {}
        self.state_preferences = {}
        self.state_price_multipliers = {}
        self.primary_state = None
        self.make_purchase_counts = {}
        self.model_purchase_counts = {}
        self.config = AnalyzerConfig() if config is None else config
        
    def analyze_purchase_patterns(self, dealer_files: List[str]) -> Dict:
        """
        Analyze all dealer purchase data to establish bidding patterns
        
        Args:
            dealer_files: List of dealer purchase history CSV files
            
        Returns:
            Dictionary containing bidding analysis results
        """
        logger.info(f"Analyzing purchase patterns from {len(dealer_files)} dealer files...")
        
        # Combine all dealer data
        all_purchases = []
        
        for file_path in dealer_files:
            try:
                df = pd.read_csv(file_path)
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Standardize price column
                if 'Sale price' in df.columns:
                    df['Sale price'] = df['Sale price'].astype(str).str.replace('$', '').str.replace(',', '')
                    df['Sale price'] = pd.to_numeric(df['Sale price'], errors='coerce')
                    
                # Filter out invalid prices
                df = df[df['Sale price'] > 0]
                all_purchases.append(df)
                logger.debug(f"Loaded {len(df)} purchases from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_purchases:
            logger.error("No valid purchase data found")
            return {}
            
        # Combine all data
        self.purchase_data = pd.concat(all_purchases, ignore_index=True)
        self._normalize_purchase_columns()
        logger.info(f"Combined {len(self.purchase_data)} total purchases")

        # Track purchase frequencies for dynamic rules
        if 'Make' in self.purchase_data.columns:
            self.make_purchase_counts = (
                self.purchase_data['Make']
                .dropna()
                .value_counts()
                .to_dict()
            )
        else:
            self.make_purchase_counts = {}

        if {'Make', 'Model'}.issubset(self.purchase_data.columns):
            self.model_purchase_counts = (
                self.purchase_data.groupby(['Make', 'Model'])['Sale price']
                .size()
                .to_dict()
            )
        else:
            self.model_purchase_counts = {}

        # Analyze different categories
        self._analyze_make_model_patterns()
        self._analyze_year_patterns()
        self._analyze_damage_patterns()
        self._analyze_location_patterns()

        # Calculate bidding rules
        self._calculate_bidding_rules()

        return {
            'total_purchases': len(self.purchase_data),
            'average_price': self.purchase_data['Sale price'].mean(),
            'median_price': self.purchase_data['Sale price'].median(),
            'price_range': (self.purchase_data['Sale price'].min(), self.purchase_data['Sale price'].max()),
            'make_model_stats': self.make_model_stats,
            'year_stats': self.year_stats,
            'damage_stats': self.damage_stats,
            'bidding_rules': self.bidding_rules
        }

    def _normalize_purchase_columns(self):
        """Normalize key purchase columns for consistent analysis."""
        if self.purchase_data is None or self.purchase_data.empty:
            return

        for column in ['Make', 'Model', 'Damage', 'Location']:
            if column in self.purchase_data.columns:
                self.purchase_data[column] = (
                    self.purchase_data[column]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

        if 'Year' in self.purchase_data.columns:
            self.purchase_data['Year'] = pd.to_numeric(self.purchase_data['Year'], errors='coerce')
            self.purchase_data = self.purchase_data[self.purchase_data['Year'].notna()]
            self.purchase_data['Year'] = self.purchase_data['Year'].astype(int)

    @staticmethod
    def _extract_state(location: str) -> str:
        """Extract the state code from a Copart location string."""
        if not location or location == 'NAN':
            return 'UNKNOWN'

        normalized = str(location).upper().strip()
        if ' - ' in normalized:
            return normalized.split(' - ')[0].strip()

        return normalized.split()[0].strip()
    
    def _analyze_make_model_patterns(self):
        """Analyze purchase patterns by make and model"""
        logger.debug("Analyzing make/model purchase patterns...")

        # Group by Make and Model
        make_stats = self.purchase_data.groupby('Make')['Sale price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        model_stats = self.purchase_data.groupby(['Make', 'Model'])['Sale price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        self.make_model_stats = {
            'make_stats': make_stats.to_dict('index'),
            'model_stats': model_stats.to_dict('index')
        }

        for make, stats in make_stats.iterrows():
            if stats['count'] >= 3:
                make_data = self.purchase_data[self.purchase_data['Make'] == make]['Sale price']
                self.make_model_stats['make_stats'][make]['percentile_75'] = float(np.percentile(make_data, 75))
                self.make_model_stats['make_stats'][make]['percentile_90'] = float(np.percentile(make_data, 90))

        percentile_models = {}
        for (make, model), stats in model_stats.iterrows():
            if stats['count'] >= 3:
                model_data = self.purchase_data[
                    (self.purchase_data['Make'] == make) &
                    (self.purchase_data['Model'] == model)
                ]['Sale price']
                percentile_models[(make, model)] = {
                    'percentile_75': float(np.percentile(model_data, 75)),
                    'percentile_90': float(np.percentile(model_data, 90))
                }

        if percentile_models:
            self.make_model_stats['model_percentiles'] = percentile_models
    
    def _analyze_year_patterns(self):
        """Analyze purchase patterns by vehicle year"""
        logger.debug("Analyzing year purchase patterns...")
        
        year_stats = self.purchase_data.groupby('Year')['Sale price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        self.year_stats = year_stats.to_dict('index')
        if not year_stats.empty:
            counts = year_stats['count'].sort_values(ascending=False)
            cumulative_share = counts.cumsum() / counts.sum()
            core_years = counts.index[cumulative_share <= 0.8].tolist()
            if not core_years:
                core_years = counts.index.tolist()

            self.year_stats['preferred_years'] = [int(year) for year in counts.index.tolist()]
            self.year_stats['core_year_band'] = {
                'min_year': int(min(core_years)),
                'max_year': int(max(core_years))
            }
    
    def _analyze_damage_patterns(self):
        """Analyze purchase patterns by damage type"""
        logger.debug("Analyzing damage type purchase patterns...")
        
        damage_stats = self.purchase_data.groupby('Damage')['Sale price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        self.damage_stats = damage_stats.to_dict('index')
        if not damage_stats.empty:
            ordered_damage = damage_stats['count'].sort_values(ascending=False)
            self.damage_stats['preferred_damage_types'] = [str(dmg) for dmg in ordered_damage.index.tolist()]
    
    def _analyze_location_patterns(self):
        """Analyze purchase patterns by location"""
        logger.debug("Analyzing location purchase patterns...")
        
        location_stats = self.purchase_data.groupby('Location')['Sale price'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        self.location_stats = location_stats.to_dict('index')
        if 'Location' in self.purchase_data.columns:
            normalized_locations = self.purchase_data['Location'].dropna().astype(str).str.upper()
            if not normalized_locations.empty:
                self.location_preferences = normalized_locations.value_counts(normalize=True).to_dict()
                state_series = normalized_locations.apply(self._extract_state)
                self.state_preferences = state_series.value_counts(normalize=True).to_dict()
                if self.state_preferences:
                    self.primary_state = max(self.state_preferences, key=self.state_preferences.get)

                state_prices = (
                    self.purchase_data.loc[state_series.index]
                    .assign(state=state_series)
                    .groupby('state')['Sale price']
                    .median()
                )
                overall_median = self.purchase_data['Sale price'].median()
                if overall_median > 0:
                    self.state_price_multipliers = (
                        state_prices / overall_median
                    ).to_dict()
    
    def _calculate_bidding_rules(self):
        """Calculate disciplined bidding rules based on purchase patterns"""
        logger.info("Calculating data-backed bidding rules...")
        
        # Overall conservative bidding rule (75th percentile)
        overall_75th = np.percentile(self.purchase_data['Sale price'], 75)
        overall_median = self.purchase_data['Sale price'].median()
        overall_90th = np.percentile(self.purchase_data['Sale price'], 90)

        self.bidding_rules = {
            'overall_percentiles': {
                'median': float(overall_median),
                'percentile_75': float(overall_75th),
                'percentile_90': float(overall_90th)
            },
            'conservative_limit': float(overall_75th),
            'moderate_limit': float(overall_median * 1.25),
            'aggressive_limit': float(overall_90th),
        }

        # Make-specific rules derived from dealer purchases
        make_specific_rules = {}
        preferred_makes = sorted(
            self.make_purchase_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for make, purchase_count in preferred_makes:
            make_data = self.purchase_data[self.purchase_data['Make'] == make]['Sale price']
            if purchase_count < 3 or make_data.empty:
                continue

            make_specific_rules[make] = {
                'conservative': float(np.percentile(make_data, 70)),
                'moderate': float(np.percentile(make_data, 80)),
                'aggressive': float(np.percentile(make_data, 90)),
                'observations': int(purchase_count)
            }

        self.bidding_rules['make_specific'] = make_specific_rules
        self.bidding_rules['preferred_makes'] = [make for make, _ in preferred_makes]

        # Model-specific rules derived from dealer purchases
        model_specific_rules = {}
        sorted_models = sorted(
            self.model_purchase_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )

        for (make, model), purchase_count in sorted_models:
            model_data = self.purchase_data[
                (self.purchase_data['Make'] == make) &
                (self.purchase_data['Model'] == model)
            ]['Sale price']

            if purchase_count < 3 or model_data.empty:
                continue

            model_specific_rules[f"{make}_{model}"] = {
                'conservative': float(np.percentile(model_data, 70)),
                'moderate': float(np.percentile(model_data, 80)),
                'aggressive': float(np.percentile(model_data, 90)),
                'observations': int(purchase_count)
            }

        self.bidding_rules['model_specific'] = model_specific_rules
    
    def calculate_bid_limit(self, make: str, model: str, year: int, damage: str, 
                          odometer: int = None, location: str = None, strategy: str = 'moderate') -> Tuple[int, str]:
        """
        Calculate bid limit based purely on dealer purchase patterns with intelligent variable matching
        
        This system analyzes what dealers actually paid for similar vehicles, considering:
        - Make, model, year variations
        - Mileage patterns (high/low/normal for age)
        - Damage type similarities  
        - Location preferences
        - Market timing patterns
        
        Args:
            make: Vehicle make
            model: Vehicle model  
            year: Vehicle year
            damage: Damage description
            odometer: Vehicle mileage
            location: Auction location
            strategy: Bidding strategy ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Tuple of (bid_limit, reasoning_with_match_details)
        """
        if self.purchase_data is None or self.purchase_data.empty:
            return 0, "No dealer purchase data available"
        
        # Clean and normalize inputs
        make = str(make).upper().strip()
        model = str(model).upper().strip()
        damage = str(damage).upper().strip() if damage else ""
        
        # Find similar vehicles that dealers actually purchased
        similar_purchases = self._find_intelligent_matches(make, model, year, odometer, damage, location)
        
        if similar_purchases.empty:
            # No matches found - use make-level fallback
            fallback_bid, fallback_reason = self._calculate_make_fallback(make, year, strategy)
            return fallback_bid, f"No similar purchases found. {fallback_reason}"
        
        # Analyze purchase patterns to determine optimal bid
        bid_limit, match_details = self._analyze_dealer_purchase_patterns(
            similar_purchases, make, model, year, odometer, damage, location, strategy
        )
        
        # Apply final safety caps
        final_bid = max(bid_limit, 300)   # Minimum $300
        final_bid = min(final_bid, 25000)  # Maximum $25,000

        # Contextual adjustments (mileage/location-based)
        adjustments: List[str] = []
        multiplier = 1.0
        adjust_cfg = self.config.bid_adjustments

        if odometer:
            if odometer > adjust_cfg.mileage_penalty_threshold:
                multiplier *= adjust_cfg.mileage_penalty_multiplier
                adjustments.append("high mileage discount")
            elif odometer < adjust_cfg.low_mileage_bonus_threshold:
                multiplier *= adjust_cfg.low_mileage_bonus_multiplier
                adjustments.append("low mileage bonus")

        if location:
            location_normalized = self._extract_state(str(location))
            if location_normalized in (self.location_preferences or {}):
                multiplier *= adjust_cfg.preferred_location_bonus
                adjustments.append("preferred yard bonus")
            else:
                multiplier *= adjust_cfg.non_preferred_location_penalty
                adjustments.append("non-core yard discount")
        else:
            multiplier *= adjust_cfg.distance_unknown_penalty
            adjustments.append("location unknown penalty")

        adjusted_bid = int(max(0, round(final_bid * multiplier)))
        if adjusted_bid <= 0:
            adjusted_bid = final_bid

        if adjustments:
            match_details += f" | Adjustments: {', '.join(adjustments)}"
        
        return adjusted_bid, match_details
    
    def _find_intelligent_matches(self, make: str, model: str, year: int, odometer: int, damage: str, location: str) -> pd.DataFrame:
        """Find dealer purchases with intelligent similarity matching"""
        
        if self.purchase_data is None or self.purchase_data.empty:
            return pd.DataFrame()
        
        # Start with all purchases and score by similarity
        matches = self.purchase_data.copy()
        matches['similarity_score'] = 0.0
        
        # Exact make match (required)
        make_mask = matches['Make'].str.upper() == make
        matches = matches[make_mask].copy()
        
        if matches.empty:
            return pd.DataFrame()
        
        # Score 1: Model similarity (0-40 points)
        model_exact = matches['Model'].str.upper().str.contains(model.split()[0], na=False)
        model_partial = matches['Model'].str.upper().str.contains(model[:4], na=False) if len(model) > 4 else False
        
        matches.loc[model_exact, 'similarity_score'] += 40
        matches.loc[model_partial & ~model_exact, 'similarity_score'] += 20
        
        # Score 2: Year proximity (0-25 points)
        year_diff = abs(matches['Year'] - year)
        matches.loc[year_diff == 0, 'similarity_score'] += 25
        matches.loc[year_diff == 1, 'similarity_score'] += 20
        matches.loc[year_diff == 2, 'similarity_score'] += 15
        matches.loc[year_diff == 3, 'similarity_score'] += 10
        matches.loc[year_diff <= 5, 'similarity_score'] += 5
        
        # Score 3: Mileage pattern similarity (0-20 points)
        if odometer and odometer > 0:
            current_year = 2025
            vehicle_age = current_year - year
            expected_mileage = vehicle_age * 12000
            
            if expected_mileage > 0:
                target_ratio = odometer / expected_mileage
                
                # Calculate mileage ratios for historical purchases
                matches['purchase_age'] = current_year - matches['Year']
                matches['expected_purchase_mileage'] = matches['purchase_age'] * 12000
                
                # Check if odometer column exists in dealer data
                if 'Odometer' in matches.columns:
                    valid_mileage = (matches['Odometer'] > 0) & (matches['expected_purchase_mileage'] > 0)
                    matches.loc[valid_mileage, 'mileage_ratio'] = matches.loc[valid_mileage, 'Odometer'] / matches.loc[valid_mileage, 'expected_purchase_mileage']
                else:
                    # No odometer data in dealer files - skip mileage scoring
                    matches['mileage_ratio'] = 1.0
                
                # Score based on similar mileage patterns
                ratio_diff = abs(matches['mileage_ratio'] - target_ratio)
                matches.loc[ratio_diff <= 0.2, 'similarity_score'] += 20  # Very similar mileage pattern
                matches.loc[(ratio_diff > 0.2) & (ratio_diff <= 0.5), 'similarity_score'] += 10  # Somewhat similar
                matches.loc[(ratio_diff > 0.5) & (ratio_diff <= 1.0), 'similarity_score'] += 5   # Moderately similar
        
        # Score 4: Damage type similarity (0-15 points)
        if damage:
            damage_normalized = self._normalize_damage_type(damage)
            purchase_damage_normalized = matches['Damage'].apply(self._normalize_damage_type)
            
            damage_exact = purchase_damage_normalized.str.upper() == damage_normalized.upper()
            damage_similar = purchase_damage_normalized.str.upper().str.contains(damage_normalized.upper()[:4], na=False) if len(damage_normalized) > 4 else False
            
            matches.loc[damage_exact, 'similarity_score'] += 15
            matches.loc[damage_similar & ~damage_exact, 'similarity_score'] += 8
        
        # Score 5: Location similarity (0-20 points)
        if location:
            location_scores = self._calculate_location_similarity(matches, location)
            matches['similarity_score'] += location_scores
        
        # Return top matches (similarity score >= 25 for meaningful matches with location)
        top_matches = matches[matches['similarity_score'] >= 25].copy()
        return top_matches.sort_values('similarity_score', ascending=False)
    
    def _calculate_location_similarity(self, matches: pd.DataFrame, target_location: str) -> pd.Series:
        """Calculate location similarity scores based on dealer purchase patterns"""

        if matches.empty or not target_location:
            return pd.Series(0.0, index=matches.index, dtype=float)

        location_scores = pd.Series(0.0, index=matches.index, dtype=float)
        target_location_normalized = str(target_location).upper().strip()
        target_state = self._extract_state(target_location_normalized)

        for idx, row in matches.iterrows():
            purchase_location = str(row.get('Location', '')).upper().strip()
            if not purchase_location or purchase_location == 'NAN':
                continue

            score = 0.0

            if purchase_location == target_location_normalized:
                score = 20.0
            else:
                purchase_state = self._extract_state(purchase_location)

                if target_state and purchase_state == target_state:
                    state_weight = self.state_preferences.get(purchase_state, 0.0)
                    score = 10.0 + (10.0 * state_weight)
                else:
                    location_weight = self.location_preferences.get(purchase_location, 0.0)
                    if location_weight:
                        score = max(score, 5.0 + 15.0 * location_weight)

                    if target_state and purchase_state:
                        neighbor_weight = self.state_preferences.get(purchase_state, 0.0)
                        if neighbor_weight:
                            score = max(score, 5.0 * neighbor_weight)

            location_scores.at[idx] = min(score, 20.0)

        return location_scores
    
    def _analyze_dealer_purchase_patterns(self, similar_purchases: pd.DataFrame, make: str, model: str, 
                                        year: int, odometer: int, damage: str, location: str, strategy: str) -> Tuple[int, str]:
        """Analyze dealer purchase patterns to determine optimal bid limit"""
        
        if similar_purchases.empty:
            return 0, "No similar purchases found"
        
        # Group by similarity tiers for analysis (updated for 120-point scale with location)
        high_similarity = similar_purchases[similar_purchases['similarity_score'] >= 70]
        medium_similarity = similar_purchases[(similar_purchases['similarity_score'] >= 50) & (similar_purchases['similarity_score'] < 70)]
        low_similarity = similar_purchases[(similar_purchases['similarity_score'] >= 25) & (similar_purchases['similarity_score'] < 50)]
        
        # Calculate base bid from purchase patterns
        if not high_similarity.empty:
            # Use high similarity matches as primary guide
            primary_data = high_similarity['Sale price']
            confidence_level = "HIGH"
            match_count = len(high_similarity)
        elif not medium_similarity.empty:
            # Use medium similarity matches
            primary_data = medium_similarity['Sale price']
            confidence_level = "MEDIUM"
            match_count = len(medium_similarity)
        else:
            # Use low similarity matches
            primary_data = low_similarity['Sale price']
            confidence_level = "LOW"
            match_count = len(low_similarity)
        
        # Apply strategy-based percentiles
        if strategy == 'conservative':
            base_bid = np.percentile(primary_data, 60)  # Lower end of purchase range
        elif strategy == 'aggressive':
            base_bid = np.percentile(primary_data, 85)  # Higher end for competitive bidding
        else:  # moderate
            base_bid = np.percentile(primary_data, 75)  # Upper-middle range
        
        # Apply mileage adjustments based on dealer patterns
        if odometer and odometer > 0:
            mileage_adjustment = self._calculate_mileage_adjustment_from_patterns(
                similar_purchases, year, odometer
            )
            base_bid *= mileage_adjustment
        
        # Apply location-based price adjustments
        location_adjustment = self._calculate_location_price_adjustment(similar_purchases, location)
        base_bid *= location_adjustment
        
        # Apply distance penalty from North Carolina
        distance_adjustment = self._calculate_distance_penalty_from_nc(location)
        base_bid *= distance_adjustment
        
        # Build detailed reasoning with location and distance details
        avg_similarity = similar_purchases['similarity_score'].mean()
        price_range = f"${primary_data.min():,.0f}-${primary_data.max():,.0f}"
        
        # Location matching details
        location_matches = self._get_location_match_summary(similar_purchases, location)
        
        # Distance penalty details - extract state for display
        if location and ' - ' in location:
            state = location.split(' - ')[0].strip()
        else:
            state = 'Unknown'
        distance_penalty_pct = int((1 - distance_adjustment) * 100)
        
        reasoning = (f"Matched {match_count} dealer purchases (avg similarity: {avg_similarity:.1f}/120) | "
                    f"Range: {price_range} | Location: {location_matches} | "
                    f"Distance from NC: -{distance_penalty_pct}% ({state}) | "
                    f"Strategy: {strategy} @ {75 if strategy=='moderate' else (60 if strategy=='conservative' else 85)}th percentile | "
                    f"Confidence: {confidence_level} → ${base_bid:,.0f}")
        
        return int(base_bid), reasoning
    
    def _calculate_location_price_adjustment(self, similar_purchases: pd.DataFrame, target_location: str) -> float:
        """Calculate price adjustment based on location-specific dealer patterns"""
        
        if target_location is None or similar_purchases.empty:
            return 1.0
        
        target_location = target_location.upper()
        target_state = self._extract_state(target_location)
        
        # Group purchases by location similarity
        exact_location = similar_purchases[
            similar_purchases['Location'].str.upper() == target_location
        ]
        
        same_state = (
            similar_purchases[
                similar_purchases['Location'].str.upper().apply(self._extract_state) == target_state
            ]
            if target_state and 'Location' in similar_purchases.columns else pd.DataFrame()
        )
        
        # Calculate price adjustments based on location patterns
        if not exact_location.empty and len(exact_location) >= 3:
            # Use exact location pricing (no adjustment needed)
            return 1.0
        elif not same_state.empty and len(same_state) >= 5:
            # Compare same-state pricing to overall patterns
            same_state_median = same_state['Sale price'].median()
            overall_median = similar_purchases['Sale price'].median()
            
            if overall_median > 0:
                adjustment = same_state_median / overall_median
                # Cap adjustments to reasonable ranges
                return max(0.85, min(1.15, adjustment))
        
        # Regional adjustment based on market patterns
        regional_adjustment = self._get_regional_market_adjustment(target_state)
        return regional_adjustment

    def _get_regional_market_adjustment(self, state: str) -> float:
        """Get regional market adjustments based on dealer purchase patterns"""
        if not state or state == 'UNKNOWN':
            return 1.0

        multiplier = self.state_price_multipliers.get(state)
        if multiplier:
            return max(0.85, min(1.2, float(multiplier)))

        return 1.0
    
    def _get_location_match_summary(self, similar_purchases: pd.DataFrame, target_location: str) -> str:
        """Get summary of location matching for reasoning"""
        
        if target_location is None:
            return "no location data"
        
        target_location = target_location.upper()
        target_state = self._extract_state(target_location)
        
        exact_matches = len(similar_purchases[
            similar_purchases['Location'].str.upper() == target_location
        ])
        
        state_matches = len(similar_purchases[
            similar_purchases['Location'].str.upper().apply(self._extract_state) == target_state
        ]) if target_state else 0
        
        if exact_matches >= 3:
            return f"{exact_matches} exact location matches"
        elif state_matches >= 5:
            return f"{state_matches} same-state matches"
        else:
            return f"{state_matches} regional matches"
    
    def _calculate_distance_penalty_from_nc(self, location: str) -> float:
        """Calculate bid adjustment based on distance from North Carolina"""
        
        if location is None:
            return 1.0
        
        location = location.upper()

        target_state = self._extract_state(location)
        if not self.primary_state:
            return 1.0

        if target_state == self.primary_state:
            return 1.0

        if not self.state_preferences:
            return 0.9

        max_weight = max(self.state_preferences.values()) if self.state_preferences else 0.0
        state_weight = self.state_preferences.get(target_state, 0.0)

        if max_weight <= 0:
            return 0.9

        normalized = state_weight / max_weight
        penalty = 0.85 + (0.15 * normalized)

        return max(0.75, min(1.0, penalty))
    
    def _calculate_mileage_adjustment_from_patterns(self, similar_purchases: pd.DataFrame, year: int, odometer: int) -> float:
        """Calculate mileage adjustment based on actual dealer purchase patterns"""
        
        current_year = 2025
        vehicle_age = current_year - year
        expected_mileage = vehicle_age * 12000
        
        if expected_mileage <= 0:
            return 1.0
        
        target_ratio = odometer / expected_mileage
        
        # Analyze how dealers valued different mileage ratios
        if 'Odometer' not in similar_purchases.columns:
            return 1.0  # No odometer data available in dealer files
            
        valid_purchases = similar_purchases[
            (similar_purchases['Odometer'] > 0) & 
            (similar_purchases['Year'] > 0)
        ].copy()
        
        if valid_purchases.empty:
            return 1.0
        
        # Calculate mileage ratios for historical purchases
        valid_purchases['purchase_age'] = current_year - valid_purchases['Year']
        valid_purchases['expected_mileage'] = valid_purchases['purchase_age'] * 12000
        valid_purchases = valid_purchases[valid_purchases['expected_mileage'] > 0]
        valid_purchases['mileage_ratio'] = valid_purchases['Odometer'] / valid_purchases['expected_mileage']
        
        # Find purchases with similar mileage patterns
        similar_mileage = valid_purchases[
            abs(valid_purchases['mileage_ratio'] - target_ratio) <= 0.3
        ]
        
        if len(similar_mileage) >= 3:
            # Use actual dealer patterns for mileage adjustment
            normal_mileage = valid_purchases[
                (valid_purchases['mileage_ratio'] >= 0.8) & 
                (valid_purchases['mileage_ratio'] <= 1.2)
            ]
            
            if not normal_mileage.empty and not similar_mileage.empty:
                normal_avg = normal_mileage['Sale price'].median()
                similar_avg = similar_mileage['Sale price'].median()
                
                if normal_avg > 0:
                    return similar_avg / normal_avg
        
        # Conservative fallback adjustments
        if target_ratio > 2.0:    # Very high mileage
            return 0.85
        elif target_ratio > 1.5:  # High mileage  
            return 0.92
        elif target_ratio < 0.5:  # Very low mileage
            return 1.08
        else:
            return 1.0
    
    def _calculate_make_fallback(self, make: str, year: int, strategy: str) -> Tuple[int, str]:
        """Calculate fallback bid when no similar vehicles found"""
        
        make_data = self.purchase_data[
            self.purchase_data['Make'].str.upper() == make
        ]['Sale price']
        
        if make_data.empty:
            return 500, f"No {make} purchases found - using minimum fallback"
        
        if strategy == 'conservative':
            fallback = np.percentile(make_data, 40)
        elif strategy == 'aggressive':
            fallback = np.percentile(make_data, 70)
        else:
            fallback = np.percentile(make_data, 55)
        
        # Age adjustment
        current_year = 2025
        age = current_year - year
        if age > 10:
            fallback *= 0.8
        elif age > 15:
            fallback *= 0.6
        
        reason = f"Using {make} average from {len(make_data)} purchases, adjusted for {age}-year age"
        return int(fallback), reason
    
    def _find_vehicle_matches(self, make: str, model: str, year: int, damage: str, location: str) -> Dict:
        """Find matching vehicles in purchase data with different levels of specificity"""
        
        # Normalize damage types for better matching
        damage_normalized = self._normalize_damage_type(damage)
        
        # Level 1: Exact matches (make, model, year ±1, similar damage)
        exact_matches = self.purchase_data[
            (self.purchase_data['Make'].str.upper() == make) &
            (self.purchase_data['Model'].str.upper().str.contains(model.split()[0], na=False)) &
            (abs(self.purchase_data['Year'] - year) <= 1) &
            (self.purchase_data['Damage'].str.upper().str.contains(damage_normalized, na=False))
        ].copy()
        
        # Level 2: Model + Year matches (same make, model, year ±2)
        model_year_matches = self.purchase_data[
            (self.purchase_data['Make'].str.upper() == make) &
            (self.purchase_data['Model'].str.upper().str.contains(model.split()[0], na=False)) &
            (abs(self.purchase_data['Year'] - year) <= 2)
        ].copy()
        
        # Level 3: Model matches (same make, model, any year)
        model_matches = self.purchase_data[
            (self.purchase_data['Make'].str.upper() == make) &
            (self.purchase_data['Model'].str.upper().str.contains(model.split()[0], na=False))
        ].copy()
        
        # Level 4: Make + Year matches (same make, year ±3)
        make_year_matches = self.purchase_data[
            (self.purchase_data['Make'].str.upper() == make) &
            (abs(self.purchase_data['Year'] - year) <= 3)
        ].copy()
        
        # Level 5: Make matches (same make, any year/model)
        make_matches = self.purchase_data[
            self.purchase_data['Make'].str.upper() == make
        ].copy()
        
        return {
            'exact_matches': exact_matches,
            'model_year_matches': model_year_matches,
            'model_matches': model_matches,
            'make_year_matches': make_year_matches,
            'make_matches': make_matches
        }
    
    def _normalize_damage_type(self, damage: str) -> str:
        """Normalize damage descriptions for better matching"""
        if not damage:
            return ""
        
        damage_upper = damage.upper()
        
        # Group similar damage types
        if any(term in damage_upper for term in ['FRONT', 'FRONT END']):
            return 'FRONT'
        elif any(term in damage_upper for term in ['REAR', 'REAR END']):
            return 'REAR'
        elif any(term in damage_upper for term in ['SIDE', 'LEFT', 'RIGHT']):
            return 'SIDE'
        elif any(term in damage_upper for term in ['HAIL', 'WEATHER']):
            return 'HAIL'
        elif any(term in damage_upper for term in ['MINOR', 'SCRATCH']):
            return 'MINOR'
        else:
            return damage_upper
    
    def _calculate_price_from_matches(self, matches: pd.DataFrame, match_type: str) -> Tuple[float, str]:
        """Calculate price statistics from matching purchases"""
        if matches.empty:
            return 0, f"No {match_type}s found"
        
        prices = matches['Sale price'].dropna()
        if prices.empty:
            return 0, f"No valid prices in {match_type}s"
        
        # Remove extreme outliers (beyond 3 standard deviations)
        if len(prices) > 3:
            mean_price = prices.mean()
            std_price = prices.std()
            prices = prices[abs(prices - mean_price) <= 3 * std_price]
        
        if prices.empty:
            return 0, f"No valid prices after outlier removal in {match_type}s"
        
        # Calculate statistics
        median_price = prices.median()
        mean_price = prices.mean()
        count = len(prices)
        
        # Use median for robustness, but weight toward mean if we have enough data
        if count >= 5:
            # Weighted average favoring median but incorporating mean
            base_price = median_price * 0.7 + mean_price * 0.3
            confidence = "high"
        elif count >= 3:
            base_price = median_price
            confidence = "medium"
        else:
            base_price = median_price
            confidence = "low"
        
        reasoning = f"{match_type} ({count} sales, {confidence} confidence): ${base_price:,.0f}"
        
        return base_price, reasoning
    
    def _apply_mileage_adjustment(self, base_price: float, odometer: int, year: int) -> Tuple[float, str]:
        """Apply mileage-based price adjustments"""
        if not odometer or odometer <= 0:
            return base_price, "no mileage data"
        
        # Calculate expected mileage based on age (average 12k miles/year)
        current_year = 2025
        vehicle_age = current_year - year
        expected_mileage = vehicle_age * 12000
        
        if expected_mileage <= 0:
            return base_price, "new vehicle"
        
        # Calculate mileage ratio
        mileage_ratio = odometer / expected_mileage
        
        # Apply adjustments
        if mileage_ratio > 2.0:  # Very high mileage
            multiplier = 0.75
            reason = f"very high miles ({odometer:,}) -25%"
        elif mileage_ratio > 1.5:  # High mileage
            multiplier = 0.85
            reason = f"high miles ({odometer:,}) -15%"
        elif mileage_ratio > 1.2:  # Above average
            multiplier = 0.95
            reason = f"above avg miles ({odometer:,}) -5%"
        elif mileage_ratio < 0.3:  # Very low mileage
            multiplier = 1.15
            reason = f"very low miles ({odometer:,}) +15%"
        elif mileage_ratio < 0.6:  # Low mileage
            multiplier = 1.08
            reason = f"low miles ({odometer:,}) +8%"
        else:  # Average mileage
            multiplier = 1.0
            reason = f"avg miles ({odometer:,})"
        
        adjusted_price = base_price * multiplier
        return adjusted_price, reason
    
    def _get_fallback_price(self, make: str, year: int) -> Tuple[float, str]:
        """Get fallback price when no specific matches found"""
        
        if make in self.make_purchase_counts and self.make_purchase_counts[make] >= 3:
            make_data = self.purchase_data[
                self.purchase_data['Make'] == make
            ]['Sale price'].dropna()

            if not make_data.empty:
                fallback_price = make_data.median()
                return fallback_price, (
                    f"{make} median from {self.make_purchase_counts[make]} purchases: ${fallback_price:,.0f}"
                )

        # Calculate year-based adjustment on global median
        global_median = self.purchase_data['Sale price'].median()
        
        # Adjust for vehicle age
        current_year = 2025
        vehicle_age = current_year - year
        
        if vehicle_age <= 3:  # Recent vehicles
            multiplier = 1.3
            age_desc = "recent"
        elif vehicle_age <= 7:  # Mid-age vehicles
            multiplier = 1.1
            age_desc = "mid-age"
        elif vehicle_age <= 12:  # Older vehicles
            multiplier = 0.9
            age_desc = "older"
        else:  # Very old vehicles
            multiplier = 0.7
            age_desc = "very old"
        
        fallback_price = global_median * multiplier
        return fallback_price, f"global median + {age_desc} adj: ${fallback_price:,.0f}"
    
    def _estimate_retail_value(self, make: str, model: str, year: int, odometer: int = None) -> float:
        """Estimate retail/resale value based on market data and vehicle characteristics"""
        
        # Base retail values by make/model (conservative estimates)
        retail_values = {
            'HONDA': {
                'ACCORD': {'base': 18000, 'premium': 25000},
                'CIVIC': {'base': 15000, 'premium': 22000},
                'CRV': {'base': 20000, 'premium': 28000},
                'PILOT': {'base': 25000, 'premium': 35000},
                'ODYSSEY': {'base': 22000, 'premium': 32000}
            },
            'NISSAN': {
                'ALTIMA': {'base': 14000, 'premium': 20000},
                'SENTRA': {'base': 12000, 'premium': 18000},
                'MAXIMA': {'base': 16000, 'premium': 24000},
                'ROGUE': {'base': 18000, 'premium': 26000},
                'PATHFINDER': {'base': 22000, 'premium': 30000}
            },
            'INFINITI': {
                'G37': {'base': 12000, 'premium': 18000},
                'Q50': {'base': 20000, 'premium': 30000},
                'QX60': {'base': 25000, 'premium': 35000}
            },
            'ACURA': {
                'TSX': {'base': 10000, 'premium': 16000},
                'TL': {'base': 12000, 'premium': 18000},
                'MDX': {'base': 22000, 'premium': 32000},
                'RDX': {'base': 18000, 'premium': 26000}
            },
            'LEXUS': {
                'ES': {'base': 18000, 'premium': 28000},
                'IS': {'base': 16000, 'premium': 25000},
                'RX': {'base': 25000, 'premium': 38000},
                'GX': {'base': 35000, 'premium': 50000}
            },
            'TOYOTA': {
                'CAMRY': {'base': 16000, 'premium': 24000},
                'COROLLA': {'base': 14000, 'premium': 20000},
                'RAV4': {'base': 22000, 'premium': 30000},
                'HIGHLANDER': {'base': 28000, 'premium': 38000}
            }
        }
        
        # Get base value
        make_data = retail_values.get(make, {})
        model_key = model.split()[0]  # Use first word of model
        model_data = make_data.get(model_key, {'base': 8000, 'premium': 15000})
        
        base_value = model_data['base']
        premium_value = model_data['premium']
        
        # Adjust for year (depreciation)
        current_year = 2025
        age = current_year - year
        
        if age <= 2:  # Very new
            retail_value = premium_value * 0.95
        elif age <= 4:  # Recent
            retail_value = premium_value * 0.85
        elif age <= 7:  # Mid-age
            retail_value = base_value * 1.1
        elif age <= 10:  # Older
            retail_value = base_value * 0.9
        else:  # Very old
            retail_value = base_value * 0.7
        
        # Minor mileage adjustment (much smaller than before)
        if odometer:
            expected_mileage = age * 12000
            if expected_mileage > 0:
                mileage_ratio = odometer / expected_mileage
                if mileage_ratio > 2.0:  # Very high mileage
                    retail_value *= 0.92
                elif mileage_ratio > 1.5:  # High mileage  
                    retail_value *= 0.96
                elif mileage_ratio < 0.5:  # Very low mileage
                    retail_value *= 1.05
        
        return max(retail_value, 4000)  # Minimum retail value
    
    def _estimate_repair_costs(self, damage: str, make: str, year: int, retail_value: float) -> float:
        """Estimate repair costs based on damage type and vehicle characteristics"""
        
        damage_upper = damage.upper()
        current_year = 2025
        age = current_year - year
        
        # Base repair costs by damage type
        if any(term in damage_upper for term in ['FRONT', 'FRONT END']):
            base_cost = 2500
            if 'MINOR' in damage_upper:
                base_cost = 1200
        elif any(term in damage_upper for term in ['REAR', 'REAR END']):
            base_cost = 2200
            if 'MINOR' in damage_upper:
                base_cost = 1000
        elif any(term in damage_upper for term in ['SIDE', 'LEFT', 'RIGHT']):
            base_cost = 2800
            if 'MINOR' in damage_upper:
                base_cost = 1500
        elif any(term in damage_upper for term in ['HAIL']):
            base_cost = 1800
        elif any(term in damage_upper for term in ['FLOOD', 'WATER']):
            base_cost = 4500  # Much higher for flood damage
        elif any(term in damage_upper for term in ['FIRE', 'BURN']):
            base_cost = 5500  # Very high for fire damage
        elif any(term in damage_upper for term in ['ROLLOVER', 'ROOF']):
            base_cost = 6000  # Structural damage
        else:
            base_cost = 2000  # Unknown damage
        
        # Adjust for luxury makes (higher parts/labor costs)
        luxury_makes = ['LEXUS', 'INFINITI', 'ACURA']
        if make in luxury_makes:
            base_cost *= 1.3
        
        # Adjust for vehicle age (older cars may need more work)
        if age > 10:
            base_cost *= 1.2
        elif age > 15:
            base_cost *= 1.4
        
        # Cap repair costs as percentage of retail value
        max_repair_percentage = 0.60  # Never exceed 60% of retail value
        max_repair_cost = retail_value * max_repair_percentage
        
        return min(base_cost, max_repair_cost)
    
    def _calculate_target_profit(self, retail_value: float, strategy: str) -> float:
        """Calculate target profit based on retail value and strategy"""
        
        # Profit margins by strategy
        profit_percentages = {
            'conservative': 0.25,  # 25% profit margin
            'moderate': 0.20,      # 20% profit margin  
            'aggressive': 0.15     # 15% profit margin (higher risk)
        }
        
        profit_percentage = profit_percentages.get(strategy, 0.20)
        
        # Minimum profit thresholds
        min_profit = max(1500, retail_value * 0.12)  # At least $1500 or 12%
        target_profit = retail_value * profit_percentage
        
        return max(target_profit, min_profit)
    
    def _get_historical_anchor_with_confidence(self, matches: Dict) -> Tuple[float, str]:
        """Get historical price anchor with confidence level for validation"""
        
        # Try exact matches first (highest confidence)
        if 'exact_matches' in matches and not matches['exact_matches'].empty:
            prices = matches['exact_matches']['Sale price'].dropna()
            if len(prices) >= 5:
                return np.percentile(prices, 75), 'high'
            elif len(prices) >= 3:
                return np.percentile(prices, 75), 'medium'
            elif len(prices) >= 1:
                return np.percentile(prices, 75), 'low'
        
        # Try model+year matches (medium-high confidence)
        if 'model_year_matches' in matches and not matches['model_year_matches'].empty:
            prices = matches['model_year_matches']['Sale price'].dropna()
            if len(prices) >= 8:
                return np.percentile(prices, 75), 'high'
            elif len(prices) >= 4:
                return np.percentile(prices, 75), 'medium'
            elif len(prices) >= 2:
                return np.percentile(prices, 75), 'low'
        
        # Try model matches (medium confidence)
        if 'model_matches' in matches and not matches['model_matches'].empty:
            prices = matches['model_matches']['Sale price'].dropna()
            if len(prices) >= 10:
                return np.percentile(prices, 75), 'medium'
            elif len(prices) >= 5:
                return np.percentile(prices, 75), 'low'
        
        # Try make+year matches (lower confidence)
        if 'make_year_matches' in matches and not matches['make_year_matches'].empty:
            prices = matches['make_year_matches']['Sale price'].dropna()
            if len(prices) >= 15:
                return np.percentile(prices, 75), 'low'
        
        # Try make matches (lowest confidence)
        if 'make_matches' in matches and not matches['make_matches'].empty:
            prices = matches['make_matches']['Sale price'].dropna()
            if len(prices) >= 20:
                return np.percentile(prices, 75), 'low'
        
        # If no specific matches, use global data (very low confidence)
        if self.purchase_data is not None:
            global_prices = self.purchase_data['Sale price'].dropna()
            if not global_prices.empty:
                return np.percentile(global_prices, 75), 'low'
        
        return 0, 'none'  # No historical data available
    
    def _apply_conservative_mileage_adjustment(self, base_bid: float, odometer: int, year: int) -> Tuple[float, str]:
        """Apply very conservative mileage adjustments to bid limit"""
        
        if not odometer or odometer <= 0:
            return base_bid, "no mileage"
        
        current_year = 2025
        age = current_year - year
        expected_mileage = age * 12000
        
        if expected_mileage <= 0:
            return base_bid, "new vehicle"
        
        mileage_ratio = odometer / expected_mileage
        
        # Much smaller adjustments than before
        if mileage_ratio > 2.5:  # Extremely high mileage
            multiplier = 0.95
            reason = f"extreme miles ({odometer:,}) -5%"
        elif mileage_ratio > 2.0:  # Very high mileage
            multiplier = 0.97
            reason = f"very high miles ({odometer:,}) -3%"
        elif mileage_ratio < 0.3:  # Very low mileage
            multiplier = 1.03
            reason = f"very low miles ({odometer:,}) +3%"
        else:  # Normal mileage range
            multiplier = 1.0
            reason = f"normal miles ({odometer:,})"
        
        adjusted_bid = base_bid * multiplier
        return adjusted_bid, reason
    
    def get_bidding_summary(self) -> Dict:
        """Get summary of bidding analysis for reporting"""
        if self.purchase_data is None:
            return {}

        return {
            'total_purchases_analyzed': len(self.purchase_data),
            'price_statistics': {
                'mean': self.purchase_data['Sale price'].mean(),
                'median': self.purchase_data['Sale price'].median(),
                'std': self.purchase_data['Sale price'].std(),
                'min': self.purchase_data['Sale price'].min(),
                'max': self.purchase_data['Sale price'].max()
            },
            'bidding_limits': self.bidding_rules,
            'preferred_makes_analyzed': self.bidding_rules.get('preferred_makes', []),
            'preferred_models_analyzed': list(self.bidding_rules.get('model_specific', {}).keys())
        }
