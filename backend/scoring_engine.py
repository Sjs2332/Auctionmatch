"""
Vehicle Scoring Engine

Implements intelligent scoring algorithm based on dealer purchase patterns.
"""

import logging
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from bidding_analyzer import BiddingAnalyzer
from config import AnalyzerConfig

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Scores vehicles based on dealer preference patterns"""
    
    def __init__(self,
                 dealer_patterns: Dict[str, Any],
                 bidding_analyzer: BiddingAnalyzer = None,
                 config: AnalyzerConfig = None):
        self.dealer_patterns = dealer_patterns
        self.bidding_analyzer = bidding_analyzer
        self.config = AnalyzerConfig() if config is None else config
        self.scoring_weights = {
            'make_match': 3.0,
            'model_match': 2.5,
            'year_preference': 2.0,
            'location_preference': 1.5,
            'price_fit': 2.0,
            'damage_tolerance': 1.5,
            'frequency_bonus': 1.0,
            'value_opportunity': 2.0,
            'high_signal_bonus': 5.0  # New bonus for high-signal patterns
        }

        # Dynamically derive high-signal patterns from dealer history.
        blueprint = self._build_high_signal_blueprint()
        self.high_signal_makes = blueprint['makes']
        self.high_signal_models = blueprint['models']
        self.high_signal_years = blueprint['years']
        self.high_signal_locations = blueprint['locations']

        # Cache the maximum frequency for each signal group for normalization.
        self._high_signal_max = {
            'make': max(self.high_signal_makes.values()) if self.high_signal_makes else 1.0,
            'model': max(self.high_signal_models.values()) if self.high_signal_models else 1.0,
            'year': max(self.high_signal_years.values()) if self.high_signal_years else 1.0,
            'location': max(self.high_signal_locations.values()) if self.high_signal_locations else 1.0,
        }

    def _similarity_ratio(self, first: str, second: str) -> float:
        if not first or not second:
            return 0.0
        return SequenceMatcher(None, first.upper(), second.upper()).ratio()

    def _best_weighted_similarity(self, target: str, weighted_choices: Dict[str, float], base_weight: float) -> float:
        best = 0.0
        for candidate, weight in weighted_choices.items():
            ratio = self._similarity_ratio(target, str(candidate))
            best = max(best, weight * ratio * base_weight)
        return best
        
    def score_vehicles(self, inventory: pd.DataFrame) -> pd.DataFrame:
        """
        Score all vehicles in inventory based on dealer patterns
        
        Args:
            inventory: DataFrame with filtered vehicle inventory
            
        Returns:
            DataFrame with added score and score_breakdown columns
        """
        logger.info(f"Scoring {len(inventory)} vehicles")
        
        if inventory.empty:
            return inventory
            
        # Create copy to avoid modifying original
        scored_inventory = inventory.copy()
        
        # Initialize scoring columns
        scored_inventory['score'] = 0.0
        scored_inventory['score_breakdown'] = ''
        
        # Score each vehicle
        for idx, vehicle in scored_inventory.iterrows():
            score, breakdown = self._score_individual_vehicle(vehicle)
            scored_inventory.at[idx, 'score'] = score
            scored_inventory.at[idx, 'score_breakdown'] = breakdown
            
        logger.info(f"Scoring complete. Average score: {scored_inventory['score'].mean():.2f}")
        logger.info(f"Vehicles with score ≥7: {len(scored_inventory[scored_inventory['score'] >= 7])}")
        
        # Add bidding limits if bidding analyzer is available
        if self.bidding_analyzer:
            logger.info("Calculating data-backed bidding limits...")
            bid_limits = []
            bid_reasoning = []
            
            for idx, vehicle in scored_inventory.iterrows():
                make = str(vehicle.get('Make', 'UNKNOWN')).upper()
                model = str(vehicle.get('Model Group', '') or vehicle.get('Model Detail', '') or vehicle.get('Model', ''))
                year = vehicle.get('Year', 2020)
                damage = str(vehicle.get('Damage Description', 'UNKNOWN'))
                odometer = vehicle.get('Odometer', None)
                location = str(vehicle.get('Location', ''))
                
                bid_limit, reasoning = self.bidding_analyzer.calculate_bid_limit(
                    make=make, model=model, year=year, damage=damage, 
                    odometer=odometer, location=location
                )
                
                # Ensure bid_limit is an integer
                bid_limit = int(float(bid_limit)) if bid_limit else 0
                bid_limits.append(f"${bid_limit:,}")
                bid_reasoning.append(reasoning)
            
            scored_inventory['bid_up_to'] = bid_limits
            scored_inventory['bid_reasoning'] = bid_reasoning
            
            logger.info(f"Added bidding limits ranging from {min([int(x.replace('$', '').replace(',', '')) for x in bid_limits]):,} to {max([int(x.replace('$', '').replace(',', '')) for x in bid_limits]):,}")
        
        return scored_inventory

    def _build_high_signal_blueprint(self) -> Dict[str, Dict[Any, float]]:
        """Construct a dealer-specific high-signal blueprint."""
        return {
            'makes': self._derive_high_signal_makes(),
            'models': self._derive_high_signal_models(),
            'years': self._derive_high_signal_years(),
            'locations': self._derive_high_signal_locations(),
        }

    def _derive_high_signal_makes(self, top_n: int = 5) -> Dict[str, float]:
        """Select the most purchased makes as high-signal targets."""
        make_prefs = self.dealer_patterns.get('make_preferences', {}) or {}
        if not make_prefs:
            return {}

        sorted_makes = sorted(make_prefs.items(), key=lambda item: item[1], reverse=True)
        return {make.upper(): freq for make, freq in sorted_makes[:top_n] if freq > 0}

    def _derive_high_signal_models(self, top_n: int = 5) -> Dict[str, float]:
        """Pick the strongest model signals across all makes."""
        model_prefs = self.dealer_patterns.get('model_preferences', {}) or {}
        if not model_prefs:
            return {}

        aggregated: Dict[str, float] = {}
        for models in model_prefs.values():
            for model, freq in models.items():
                model_key = str(model).upper()
                # Keep the highest observed frequency per model.
                aggregated[model_key] = max(aggregated.get(model_key, 0.0), freq)

        sorted_models = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
        return {model: freq for model, freq in sorted_models[:top_n] if freq > 0}

    def _derive_high_signal_years(self, top_n: int = 5) -> Dict[int, float]:
        """Identify the top model years favored by the dealer."""
        year_prefs = self.dealer_patterns.get('year_preferences', {}) or {}
        if not year_prefs:
            return {}

        sorted_years = sorted(year_prefs.items(), key=lambda item: item[1], reverse=True)
        return {int(year): freq for year, freq in sorted_years[:top_n] if freq > 0}

    def _derive_high_signal_locations(self, top_n: int = 10) -> Dict[str, float]:
        """Surface the most popular auction locations for the dealer."""
        location_prefs = self.dealer_patterns.get('location_preferences', {}) or {}
        if not location_prefs:
            return {}

        sorted_locations = sorted(location_prefs.items(), key=lambda item: item[1], reverse=True)
        return {location.upper(): freq for location, freq in sorted_locations[:top_n] if freq > 0}

    def _score_individual_vehicle(self, vehicle: pd.Series) -> Tuple[float, str]:
        """
        Score an individual vehicle based on dealer patterns
        
        Args:
            vehicle: Series containing vehicle data
            
        Returns:
            Tuple of (score, breakdown_explanation)
        """
        total_score = 0.0
        breakdown_parts = []
        
        # Score 1: Make preference match
        make_score = self._score_make_preference(vehicle)
        total_score += make_score
        if make_score > 0:
            breakdown_parts.append(f"Make({make_score:.1f})")
        
        # Score 2: Model preference match
        model_score = self._score_model_preference(vehicle)
        total_score += model_score
        if model_score > 0:
            breakdown_parts.append(f"Model({model_score:.1f})")
        
        # Score 3: Year preference
        year_score = self._score_year_preference(vehicle)
        total_score += year_score
        if year_score > 0:
            breakdown_parts.append(f"Year({year_score:.1f})")
        
        # Score 4: Location preference
        location_score = self._score_location_preference(vehicle)
        total_score += location_score
        if location_score > 0:
            breakdown_parts.append(f"Location({location_score:.1f})")
        
        # Score 5: Price fit
        price_score = self._score_price_fit(vehicle)
        total_score += price_score
        if price_score > 0:
            breakdown_parts.append(f"Price({price_score:.1f})")
        
        # Score 6: Damage tolerance
        damage_score = self._score_damage_tolerance(vehicle)
        total_score += damage_score
        if damage_score > 0:
            breakdown_parts.append(f"Damage({damage_score:.1f})")
        
        # Score 7: Purchase frequency bonus
        frequency_score = self._score_frequency_bonus(vehicle)
        total_score += frequency_score
        if frequency_score > 0:
            breakdown_parts.append(f"Frequency({frequency_score:.1f})")
        
        # Score 8: Value opportunity
        value_score = self._score_value_opportunity(vehicle)
        total_score += value_score
        if value_score > 0:
            breakdown_parts.append(f"Value({value_score:.1f})")
        
        # Score 9: High-signal pattern bonus
        signal_score = self._score_high_signal_patterns(vehicle)
        total_score += signal_score
        if signal_score > 0:
            breakdown_parts.append(f"Signal({signal_score:.1f})")
        
        breakdown = " + ".join(breakdown_parts) if breakdown_parts else "No matches"
        
        return round(total_score, 1), breakdown
    
    def _score_make_preference(self, vehicle: pd.Series) -> float:
        """Score based on dealer's make preferences"""
        make_prefs = self.dealer_patterns.get('make_preferences', {})
        
        if not make_prefs or 'Make' not in vehicle:
            return 0.0
        
        vehicle_make = str(vehicle['Make']).upper()
        base_weight = self.config.similarity.make_weight
        preference_score = self._best_weighted_similarity(vehicle_make, make_prefs, base_weight)

        max_weight = self.scoring_weights['make_match']
        return preference_score * max_weight * 10  # Scale up for meaningful scores
    
    def _score_model_preference(self, vehicle: pd.Series) -> float:
        """Score based on dealer's model preferences within make"""
        model_prefs = self.dealer_patterns.get('model_preferences', {})
        
        if not model_prefs or 'Make' not in vehicle or 'Model Group' not in vehicle:
            return 0.0
        
        vehicle_make = str(vehicle['Make']).upper()
        vehicle_model = str(vehicle['Model Group']).upper()
        
        make_models = model_prefs.get(vehicle_make, {})
        base_weight = self.config.similarity.model_weight
        preference_score = self._best_weighted_similarity(vehicle_model, make_models, base_weight)
        
        max_weight = self.scoring_weights['model_match']
        return preference_score * max_weight * 10
    
    def _score_year_preference(self, vehicle: pd.Series) -> float:
        """Score based on dealer's year preferences"""
        year_prefs = self.dealer_patterns.get('year_preferences', {})
        
        if not year_prefs or 'Year' not in vehicle:
            return 0.0
        
        try:
            vehicle_year = int(vehicle['Year'])
            base_weight = self.config.similarity.year_weight
            best_score = 0.0
            decay = self.config.similarity.year_decay_per_step

            for preferred_year, freq in year_prefs.items():
                diff = abs(vehicle_year - int(preferred_year))
                similarity = max(0.0, 1 - diff * decay)
                best_score = max(best_score, freq * similarity * base_weight)
            
            max_weight = self.scoring_weights['year_preference']
            return best_score * max_weight * 10
            
        except (ValueError, TypeError):
            return 0.0
    
    def _score_location_preference(self, vehicle: pd.Series) -> float:
        """Score based on dealer's location preferences"""
        location_prefs = self.dealer_patterns.get('location_preferences', {})
        
        if not location_prefs:
            return 0.0
        
        # Try multiple location column names
        location_columns = ['Yard name', 'Location', 'Yard Name']
        vehicle_location = None
        
        for col in location_columns:
            if col in vehicle and pd.notna(vehicle[col]):
                vehicle_location = str(vehicle[col]).upper()
                break
        
        if not vehicle_location:
            return 0.0
        
        # Check for partial matches (e.g., state codes)
        best_match_score = 0.0
        base_weight = self.config.similarity.location_weight
        for pref_location, score in location_prefs.items():
            similarity = self._similarity_ratio(vehicle_location, pref_location)
            best_match_score = max(best_match_score, score * similarity * base_weight)
        
        max_weight = self.scoring_weights['location_preference']
        return best_match_score * max_weight * 10
    
    def _score_price_fit(self, vehicle: pd.Series) -> float:
        """Score based on how well vehicle price fits dealer patterns"""
        price_patterns = self.dealer_patterns.get('price_patterns', {})
        
        # Find the correct high bid column
        high_bid_col = None
        for col in ['High Bid', 'High Bid =non-vix,Sealed=Vix']:
            if col in vehicle and pd.notna(vehicle[col]):
                high_bid_col = col
                break
        
        if not price_patterns or not high_bid_col:
            return 0.0
        
        try:
            vehicle_price = float(vehicle[high_bid_col])
            if vehicle_price <= 0:
                return 0.0
            
            avg_price = price_patterns.get('avg_price', 0)
            if avg_price <= 0:
                return 0.0
            
            # Score based on how close to dealer's average price
            price_ratio = vehicle_price / avg_price
            
            # Optimal range is 0.5x to 2x dealer's average
            if 0.5 <= price_ratio <= 2.0:
                # Best score for prices close to dealer average
                if 0.8 <= price_ratio <= 1.2:
                    score_multiplier = 1.0
                else:
                    score_multiplier = 0.7
            else:
                score_multiplier = 0.3
            
            max_weight = self.scoring_weights['price_fit']
            return score_multiplier * max_weight
            
        except (ValueError, TypeError):
            return 0.0
    
    def _score_damage_tolerance(self, vehicle: pd.Series) -> float:
        """Score based on dealer's damage tolerance"""
        damage_tolerance = self.dealer_patterns.get('damage_tolerance', {})
        
        if not damage_tolerance or 'Damage Description' not in vehicle:
            return 0.0
        
        vehicle_damage = str(vehicle['Damage Description']).upper()
        
        # Check for exact or partial damage type matches
        best_tolerance = 0.0
        base_weight = self.config.similarity.damage_weight
        for damage_type, tolerance in damage_tolerance.items():
            similarity = self._similarity_ratio(vehicle_damage, damage_type)
            best_tolerance = max(best_tolerance, tolerance * similarity * base_weight)
        
        max_weight = self.scoring_weights['damage_tolerance']
        return best_tolerance * max_weight * 10
    
    def _score_frequency_bonus(self, vehicle: pd.Series) -> float:
        """Bonus score based on dealer's purchase frequency"""
        frequency_data = self.dealer_patterns.get('purchase_frequency', {})
        
        if not frequency_data:
            return 0.0
        
        # Higher frequency dealers get higher confidence scores
        avg_purchases_per_month = frequency_data.get('avg_purchases_per_month', 0)
        
        if avg_purchases_per_month >= 5:  # Active dealer
            multiplier = 1.0
        elif avg_purchases_per_month >= 2:  # Moderate dealer
            multiplier = 0.7
        elif avg_purchases_per_month >= 1:  # Light dealer
            multiplier = 0.4
        else:
            multiplier = 0.0
        
        max_weight = self.scoring_weights['frequency_bonus']
        return multiplier * max_weight
    
    def _score_value_opportunity(self, vehicle: pd.Series) -> float:
        """Score based on potential value opportunity"""
        # Find the correct high bid column
        high_bid_col = None
        for col in ['High Bid', 'High Bid =non-vix,Sealed=Vix']:
            if col in vehicle and pd.notna(vehicle[col]):
                high_bid_col = col
                break
        
        if not high_bid_col or 'Est. Retail Value' not in vehicle:
            return 0.0
        
        try:
            current_bid = float(vehicle[high_bid_col])
            retail_value = float(vehicle.get('Est. Retail Value', 0))
            
            if current_bid <= 0 or retail_value <= 0:
                return 0.0
            
            # Calculate potential margin
            margin_ratio = (retail_value - current_bid) / retail_value
            
            # Score based on margin opportunity
            if margin_ratio >= 0.6:  # 60%+ margin
                score_multiplier = 1.0
            elif margin_ratio >= 0.4:  # 40-60% margin
                score_multiplier = 0.8
            elif margin_ratio >= 0.2:  # 20-40% margin
                score_multiplier = 0.6
            else:
                score_multiplier = 0.0
            
            max_weight = self.scoring_weights['value_opportunity']
            return score_multiplier * max_weight
            
        except (ValueError, TypeError):
            return 0.0
    
    def _score_high_signal_patterns(self, vehicle: pd.Series) -> float:
        """Score based on high-signal dealer blueprint patterns"""
        total_bonus = 0.0
        
        # High-signal make bonus (+2 points as per blueprint)
        if 'Make' in vehicle:
            vehicle_make = str(vehicle['Make']).upper()
            if vehicle_make in self.high_signal_makes:
                purchase_intensity = self.high_signal_makes[vehicle_make]
                max_intensity = self._high_signal_max.get('make', 1.0) or 1.0
                normalized = min(purchase_intensity / max_intensity, 1.0)
                make_bonus = 2.0 * min(normalized * 1.5, 1.5)  # Cap at 3.0
                total_bonus += make_bonus
        
        # High-signal model bonus (+2 points for exact matches)
        vehicle_model = str(vehicle.get('Model Group') or vehicle.get('Model Detail') or vehicle.get('Model', '')).upper()
        if vehicle_model:
            for signal_model, intensity in self.high_signal_models.items():
                if signal_model in vehicle_model:
                    max_intensity = self._high_signal_max.get('model', 1.0) or 1.0
                    normalized = min(intensity / max_intensity, 1.0)
                    model_bonus = 2.0 * min(normalized * 1.5, 1.5)
                    total_bonus += model_bonus
                    break
        
        # High-signal year bonus (sweet spot defined by dealer history)
        if 'Year' in vehicle:
            try:
                vehicle_year = int(vehicle['Year'])
                if vehicle_year in self.high_signal_years:
                    year_intensity = self.high_signal_years[vehicle_year]
                    max_intensity = self._high_signal_max.get('year', 1.0) or 1.0
                    normalized = min(year_intensity / max_intensity, 1.0)
                    year_bonus = 1.0 * min(normalized * 1.2, 1.2)
                    total_bonus += year_bonus
            except (ValueError, TypeError):
                pass
        
        # High-signal location bonus (+1 point for target yards)
        location_columns = ['Yard name', 'Location', 'Yard Name']
        for col in location_columns:
            if col in vehicle and pd.notna(vehicle[col]):
                vehicle_location = str(vehicle[col]).upper()
                for signal_location, intensity in self.high_signal_locations.items():
                    if signal_location.upper() in vehicle_location:
                        max_intensity = self._high_signal_max.get('location', 1.0) or 1.0
                        normalized = min(intensity / max_intensity, 1.0)
                        location_bonus = 1.0 * min(normalized * 1.2, 1.2)
                        total_bonus += location_bonus
                        break
                break
        
        return round(total_bonus, 1)
    
    def get_scoring_summary(self) -> Dict[str, Any]:
        """Get summary of scoring methodology and weights"""
        return {
            'scoring_weights': self.scoring_weights,
            'dealer_patterns_summary': {
                'makes_analyzed': len(self.dealer_patterns.get('make_preferences', {})),
                'models_analyzed': sum(len(models) for models in self.dealer_patterns.get('model_preferences', {}).values()),
                'years_analyzed': len(self.dealer_patterns.get('year_preferences', {})),
                'locations_analyzed': len(self.dealer_patterns.get('location_preferences', {})),
                'damage_types_analyzed': len(self.dealer_patterns.get('damage_tolerance', {}))
            },
            'high_signal_blueprint': {
                'makes': self.high_signal_makes,
                'models': self.high_signal_models,
                'years': self.high_signal_years,
                'locations': self.high_signal_locations
            }
        }
