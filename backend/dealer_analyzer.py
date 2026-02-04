"""
Dealer Pattern Analysis Module

Analyzes historical dealer purchase data to identify patterns and preferences.
"""

import logging
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class DealerAnalyzer:
    """Analyzes dealer purchase patterns from historical data"""
    
    def __init__(self):
        self.dealer_patterns = {}
        
    def analyze_patterns(self, dealer_files: List[str]) -> Dict[str, Any]:
        """
        Analyze dealer purchase patterns from multiple CSV files
        
        Args:
            dealer_files: List of CSV file paths containing dealer data
            
        Returns:
            Dictionary containing analyzed dealer patterns
        """
        logger.info(f"Analyzing patterns from {len(dealer_files)} dealer files")
        
        # Load and consolidate dealer data
        all_dealer_data = self._load_dealer_data(dealer_files)
        
        if all_dealer_data.empty:
            logger.error("No dealer data loaded")
            return {}
            
        # Clean and standardize data
        cleaned_data = self._clean_dealer_data(all_dealer_data)
        
        # Extract patterns
        patterns = {
            'make_preferences': self._analyze_make_preferences(cleaned_data),
            'model_preferences': self._analyze_model_preferences(cleaned_data),
            'year_preferences': self._analyze_year_preferences(cleaned_data),
            'location_preferences': self._analyze_location_preferences(cleaned_data),
            'price_patterns': self._analyze_price_patterns(cleaned_data),
            'damage_tolerance': self._analyze_damage_tolerance(cleaned_data),
            'purchase_frequency': self._analyze_purchase_frequency(cleaned_data),
            'seasonal_patterns': self._analyze_seasonal_patterns(cleaned_data)
        }
        
        logger.info("Pattern analysis complete")
        return patterns
    
    def _load_dealer_data(self, dealer_files: List[str]) -> pd.DataFrame:
        """Load and combine dealer data from multiple CSV files"""
        all_data = []
        
        for file_path in dealer_files:
            try:
                logger.debug(f"Loading dealer data from {file_path}")
                
                # Read CSV with flexible encoding handling
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin-1')
                
                if not df.empty:
                    # Add source file identifier
                    df['source_file'] = file_path
                    all_data.append(df)
                    logger.debug(f"Loaded {len(df)} records from {file_path}")
                else:
                    logger.warning(f"No data found in {file_path}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                continue
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dealer data: {len(combined_data)} total records")
            return combined_data
        else:
            logger.error("No dealer data could be loaded")
            return pd.DataFrame()
    
    def _clean_dealer_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dealer data"""
        logger.debug("Cleaning dealer data")
        
        # Remove BOM characters if present
        if 'Lot #' in data.columns:
            data = data.rename(columns={'Lot #': 'Lot_Number'})
        elif '﻿Lot #' in data.columns:
            data = data.rename(columns={'﻿Lot #': 'Lot_Number'})
        
        # Standardize column names
        column_mapping = {
            'Sale price': 'Sale_Price',
            'Payment status': 'Payment_Status',
            'Sale date': 'Sale_Date'
        }
        data = data.rename(columns=column_mapping)
        
        # Clean price data
        if 'Sale_Price' in data.columns:
            data['Sale_Price'] = data['Sale_Price'].astype(str)
            data['Sale_Price'] = data['Sale_Price'].str.replace('$', '')
            data['Sale_Price'] = data['Sale_Price'].str.replace(',', '')
            data['Sale_Price'] = pd.to_numeric(data['Sale_Price'], errors='coerce')
        
        # Filter out non-paid purchases for pattern analysis
        if 'Payment_Status' in data.columns:
            paid_data = data[data['Payment_Status'] == 'Paid'].copy()
            logger.debug(f"Filtered to {len(paid_data)} paid purchases from {len(data)} total records")
            return paid_data
        
        return data
    
    def _analyze_make_preferences(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze dealer preferences for vehicle makes"""
        if 'Make' not in data.columns:
            return {}
            
        make_counts = Counter(data['Make'].dropna())
        total_purchases = sum(make_counts.values())
        
        # Calculate preference scores (frequency-based)
        preferences = {}
        for make, count in make_counts.items():
            preferences[make.upper()] = count / total_purchases
            
        logger.debug(f"Make preferences: {len(preferences)} makes analyzed")
        return preferences
    
    def _analyze_model_preferences(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze dealer preferences for vehicle models by make"""
        if 'Make' not in data.columns or 'Model' not in data.columns:
            return {}
            
        model_preferences = defaultdict(dict)
        
        # Group by make and analyze model preferences within each make
        for make in data['Make'].dropna().unique():
            make_data = data[data['Make'] == make]
            model_counts = Counter(make_data['Model'].dropna())
            total_make_purchases = sum(model_counts.values())
            
            for model, count in model_counts.items():
                model_preferences[make.upper()][model.upper()] = count / total_make_purchases
                
        logger.debug(f"Model preferences: {len(model_preferences)} makes with model data")
        return dict(model_preferences)
    
    def _analyze_year_preferences(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze dealer preferences for vehicle years"""
        if 'Year' not in data.columns:
            return {}
            
        # Convert years to numeric and filter reasonable range
        years = pd.to_numeric(data['Year'], errors='coerce')
        years = years[(years >= 1990) & (years <= 2025)]
        
        year_counts = Counter(years.dropna())
        total_purchases = sum(year_counts.values())
        
        preferences = {}
        for year, count in year_counts.items():
            preferences[int(year)] = count / total_purchases
            
        logger.debug(f"Year preferences: {len(preferences)} years analyzed")
        return preferences
    
    def _analyze_location_preferences(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze dealer preferences for auction locations"""
        if 'Location' not in data.columns:
            return {}
            
        location_counts = Counter(data['Location'].dropna())
        total_purchases = sum(location_counts.values())
        
        preferences = {}
        for location, count in location_counts.items():
            preferences[location.upper()] = count / total_purchases
            
        logger.debug(f"Location preferences: {len(preferences)} locations analyzed")
        return preferences
    
    def _analyze_price_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dealer price patterns and ranges"""
        if 'Sale_Price' not in data.columns:
            return {}
            
        prices = data['Sale_Price'].dropna()
        
        if len(prices) == 0:
            return {}
            
        patterns = {
            'avg_price': float(prices.mean()),
            'median_price': float(prices.median()),
            'price_ranges': {
                'under_2k': len(prices[prices < 2000]) / len(prices),
                '2k_5k': len(prices[(prices >= 2000) & (prices < 5000)]) / len(prices),
                '5k_10k': len(prices[(prices >= 5000) & (prices < 10000)]) / len(prices),
                '10k_20k': len(prices[(prices >= 10000) & (prices < 20000)]) / len(prices),
                'over_20k': len(prices[prices >= 20000]) / len(prices)
            }
        }
        
        logger.debug(f"Price patterns: avg=${patterns['avg_price']:.0f}, median=${patterns['median_price']:.0f}")
        return patterns
    
    def _analyze_damage_tolerance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze dealer tolerance for different damage types"""
        if 'Damage' not in data.columns:
            return {}
            
        damage_counts = Counter(data['Damage'].dropna())
        total_purchases = sum(damage_counts.values())
        
        tolerance = {}
        for damage_type, count in damage_counts.items():
            tolerance[damage_type.upper()] = count / total_purchases
            
        logger.debug(f"Damage tolerance: {len(tolerance)} damage types analyzed")
        return tolerance
    
    def _analyze_purchase_frequency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dealer purchase frequency patterns"""
        if 'Sale_Date' not in data.columns:
            return {}
            
        # Convert sale dates and analyze frequency
        try:
            dates = pd.to_datetime(data['Sale_Date'], errors='coerce')
            dates = dates.dropna()
            
            if len(dates) == 0:
                return {}
                
            frequency = {
                'total_purchases': len(dates),
                'date_range_days': (dates.max() - dates.min()).days,
                'avg_purchases_per_month': len(dates) / max(1, (dates.max() - dates.min()).days / 30)
            }
            
            logger.debug(f"Purchase frequency: {frequency['total_purchases']} purchases over {frequency['date_range_days']} days")
            return frequency
            
        except Exception as e:
            logger.warning(f"Error analyzing purchase frequency: {str(e)}")
            return {}
    
    def _analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze seasonal purchase patterns"""
        if 'Sale_Date' not in data.columns:
            return {}
            
        try:
            dates = pd.to_datetime(data['Sale_Date'], errors='coerce')
            dates = dates.dropna()
            
            if len(dates) == 0:
                return {}
                
            # Extract months and analyze seasonal patterns
            months = dates.dt.month
            month_counts = Counter(months)
            total_purchases = sum(month_counts.values())
            
            seasonal_patterns = {}
            for month, count in month_counts.items():
                seasonal_patterns[f"month_{month}"] = count / total_purchases
                
            logger.debug(f"Seasonal patterns: {len(seasonal_patterns)} months analyzed")
            return seasonal_patterns
            
        except Exception as e:
            logger.warning(f"Error analyzing seasonal patterns: {str(e)}")
            return {}
