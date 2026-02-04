"""
Utility Functions

Helper functions for data validation, logging, and output formatting.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

DEFAULT_INVENTORY_FILENAME = "salesdata.csv"


def get_default_inventory_file() -> Optional[str]:
    """
    Resolve the default inventory CSV packaged with the service.

    Order of precedence:
      1. Explicit override via AUCTION_INVENTORY_PATH env var.
      2. Bundled `salesdata.csv` in the project root.

    Returns:
        Absolute path to the inventory CSV if it exists, otherwise None.
    """
    repo_root = Path(__file__).resolve().parent
    override = os.environ.get("AUCTION_INVENTORY_PATH")

    candidates: List[Path] = []
    if override:
        override_path = Path(override)
        if not override_path.is_absolute():
            override_path = repo_root / override_path
        candidates.append(override_path)

    candidates.append(repo_root / DEFAULT_INVENTORY_FILENAME)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return None


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('auction_analysis.log')
        ]
    )
    
    # Reduce pandas warnings
    logging.getLogger('pandas').setLevel(logging.WARNING)


def validate_files(file_paths: List[str]) -> bool:
    """
    Validate that all required files exist and are readable
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        True if all files are valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    all_valid = True
    
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            all_valid = False
        elif not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            all_valid = False
        elif path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            all_valid = False
        else:
            # Try to read first few lines to validate CSV format
            try:
                pd.read_csv(file_path, nrows=1)
                logger.debug(f"File validated: {file_path}")
            except Exception as e:
                logger.error(f"File validation failed for {file_path}: {str(e)}")
                all_valid = False
    
    return all_valid


def print_summary_stats(dealer_patterns: Dict[str, Any],
                       filtered_inventory: pd.DataFrame,
                       watchlist: pd.DataFrame,
                       inventory_filters: Optional[Dict[str, Any]] = None):
    """
    Print comprehensive summary statistics
    
    Args:
        dealer_patterns: Analyzed dealer patterns
        filtered_inventory: Filtered inventory data
        watchlist: Generated watchlist
    """
    print("\n" + "="*80)
    print("COPART ANALYSIS SUMMARY")
    print("="*80)
    
    # Dealer Pattern Analysis
    print("\n📊 DEALER PATTERN ANALYSIS")
    print("-" * 40)
    
    make_prefs = dealer_patterns.get('make_preferences', {})
    if make_prefs:
        top_makes = sorted(make_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 Preferred Makes:")
        for make, score in top_makes:
            print(f"  • {make}: {score:.1%}")
    
    price_patterns = dealer_patterns.get('price_patterns', {})
    if price_patterns:
        print(f"\nPrice Patterns:")
        print(f"  • Average Purchase: ${price_patterns.get('avg_price', 0):,.0f}")
        print(f"  • Median Purchase: ${price_patterns.get('median_price', 0):,.0f}")
    
    location_prefs = dealer_patterns.get('location_preferences', {})
    if location_prefs:
        top_locations = sorted(location_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\nTop Preferred Locations:")
        for location, score in top_locations:
            print(f"  • {location}: {score:.1%}")
    
    # Inventory Analysis
    print(f"\n📋 INVENTORY ANALYSIS")
    print("-" * 40)
    print(f"Total Vehicles Analyzed: {len(filtered_inventory):,}")

    if not filtered_inventory.empty:
        print(f"Year Range: {filtered_inventory['Year'].min():.0f} - {filtered_inventory['Year'].max():.0f}")
        print(f"Odometer Range: {filtered_inventory['Odometer'].min():,.0f} - {filtered_inventory['Odometer'].max():,.0f} miles")
        
        # Handle different possible column names for high bid
        high_bid_col = None
        for col in ['High Bid', 'High Bid =non-vix,Sealed=Vix']:
            if col in filtered_inventory.columns:
                high_bid_col = col
                break
        
        if high_bid_col:
            avg_bid = filtered_inventory[high_bid_col].mean()
            print(f"Average High Bid: ${avg_bid:,.0f}")
        
        # Top makes in inventory
        if 'Make' in filtered_inventory.columns:
            top_inventory_makes = filtered_inventory['Make'].value_counts().head(5)
            print(f"\nTop Makes in Inventory:")
            for make, count in top_inventory_makes.items():
                print(f"  • {make}: {count:,} vehicles")

    if inventory_filters:
        print(f"\n🧭 DYNAMIC INVENTORY FILTERS")
        print("-" * 40)
        for key, values in inventory_filters.items():
            if not values:
                continue
            readable_key = key.replace('_', ' ').title()
            preview = ', '.join(str(v) for v in list(values)[:5])
            more = '...' if len(values) > 5 else ''
            print(f"{readable_key}: {preview}{more}")
    
    # Watchlist Results
    print(f"\n🎯 WATCHLIST RESULTS")
    print("-" * 40)
    print(f"Vehicles in Watchlist: {len(watchlist):,}")
    
    if not watchlist.empty:
        print(f"Score Range: {watchlist['score'].min():.1f} - {watchlist['score'].max():.1f}")
        print(f"Average Score: {watchlist['score'].mean():.1f}")
        
        # Score distribution
        high_scores = len(watchlist[watchlist['score'] >= 8])
        med_scores = len(watchlist[(watchlist['score'] >= 7) & (watchlist['score'] < 8)])
        print(f"\nScore Distribution:")
        print(f"  • High Confidence (8.0+): {high_scores:,} vehicles")
        print(f"  • Medium Confidence (7.0-7.9): {med_scores:,} vehicles")
        
        # Value analysis
        high_bid_col = None
        for col in ['High Bid', 'High Bid =non-vix,Sealed=Vix']:
            if col in watchlist.columns:
                high_bid_col = col
                break
        
        if high_bid_col:
            total_watchlist_value = watchlist[high_bid_col].sum()
            avg_watchlist_price = watchlist[high_bid_col].mean()
            print(f"\nValue Analysis:")
            print(f"  • Total Watchlist Value: ${total_watchlist_value:,.0f}")
            print(f"  • Average Vehicle Price: ${avg_watchlist_price:,.0f}")
        
        # Top recommendations by make
        if 'Make' in watchlist.columns:
            top_watchlist_makes = watchlist['Make'].value_counts().head(3)
            print(f"\nTop Recommended Makes:")
            for make, count in top_watchlist_makes.items():
                avg_score = watchlist[watchlist['Make'] == make]['score'].mean()
                print(f"  • {make}: {count:,} vehicles (avg score: {avg_score:.1f})")
    
    # Performance Metrics
    print(f"\n⚡ PERFORMANCE METRICS")
    print("-" * 40)
    
    if len(filtered_inventory) > 0:
        conversion_rate = len(watchlist) / len(filtered_inventory) * 100
        print(f"Watchlist Conversion Rate: {conversion_rate:.1f}%")
    
    # Pattern Confidence
    frequency_data = dealer_patterns.get('purchase_frequency', {})
    if frequency_data:
        total_purchases = frequency_data.get('total_purchases', 0)
        print(f"Analysis Confidence: {_get_confidence_level(total_purchases)}")
        print(f"  • Based on {total_purchases:,} historical purchases")


def _get_confidence_level(purchase_count: int) -> str:
    """Determine confidence level based on purchase history"""
    if purchase_count >= 100:
        return "HIGH (100+ purchases)"
    elif purchase_count >= 50:
        return "MEDIUM (50+ purchases)"
    elif purchase_count >= 20:
        return "LOW (20+ purchases)"
    else:
        return "VERY LOW (<20 purchases)"


def format_currency(value: float) -> str:
    """Format numeric value as currency"""
    try:
        return f"${value:,.0f}"
    except (ValueError, TypeError):
        return "$0"


def format_mileage(value: float) -> str:
    """Format numeric value as mileage"""
    try:
        return f"{value:,.0f} miles"
    except (ValueError, TypeError):
        return "Unknown miles"


def export_detailed_report(watchlist: pd.DataFrame, 
                          dealer_patterns: Dict[str, Any],
                          output_file: str = "detailed_analysis_report.txt"):
    """
    Export detailed analysis report to text file
    
    Args:
        watchlist: Generated watchlist DataFrame
        dealer_patterns: Dealer pattern analysis results
        output_file: Output file path
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(output_file, 'w') as f:
            f.write("COPART DEALER ANALYSIS - DETAILED REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Dealer Patterns Detail
            f.write("DEALER PATTERN ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            for pattern_type, data in dealer_patterns.items():
                f.write(f"\n{pattern_type.upper()}:\n")
                if isinstance(data, dict):
                    for key, value in list(data.items())[:10]:  # Top 10
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.3f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {data}\n")
            
            # Top Watchlist Vehicles
            if not watchlist.empty:
                f.write(f"\n\nTOP WATCHLIST VEHICLES\n")
                f.write("-" * 30 + "\n")
                
                top_vehicles = watchlist.head(20)
                for idx, vehicle in top_vehicles.iterrows():
                    f.write(f"\nScore: {vehicle['score']:.1f}\n")
                    f.write(f"Vehicle: {vehicle.get('Year', 'N/A')} {vehicle.get('Make', 'N/A')} {vehicle.get('Model Group', 'N/A')}\n")
                    f.write(f"Price: {format_currency(vehicle.get('High Bid', 0))}\n")
                    f.write(f"Mileage: {format_mileage(vehicle.get('Odometer', 0))}\n")
                    f.write(f"Reasoning: {vehicle.get('score_breakdown', 'N/A')}\n")
        
        logger.info(f"Detailed report exported to {output_file}")
        
    except Exception as e:
        logger.error(f"Error exporting detailed report: {str(e)}")


def clean_currency_string(currency_str: str) -> float:
    """
    Clean currency string and convert to float
    
    Args:
        currency_str: String containing currency value
        
    Returns:
        Float value or 0.0 if conversion fails
    """
    try:
        if pd.isna(currency_str):
            return 0.0
        
        # Remove currency symbols and commas
        cleaned = str(currency_str).replace('$', '').replace(',', '').strip()
        return float(cleaned)
    except (ValueError, TypeError):
        return 0.0


def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with quality metrics
    """
    logger = logging.getLogger(__name__)
    
    quality_metrics = {
        'total_rows': len(df),
        'missing_columns': [],
        'completeness_scores': {},
        'data_types': {},
        'quality_score': 0.0
    }
    
    # Check for missing required columns
    for col in required_columns:
        if col not in df.columns:
            quality_metrics['missing_columns'].append(col)
    
    # Calculate completeness for each column
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        completeness = non_null_count / len(df) if len(df) > 0 else 0
        quality_metrics['completeness_scores'][col] = completeness
        quality_metrics['data_types'][col] = str(df[col].dtype)
    
    # Calculate overall quality score
    if quality_metrics['completeness_scores']:
        avg_completeness = sum(quality_metrics['completeness_scores'].values()) / len(quality_metrics['completeness_scores'])
        missing_penalty = len(quality_metrics['missing_columns']) * 0.1
        quality_metrics['quality_score'] = max(0, avg_completeness - missing_penalty)
    
    logger.debug(f"Data quality score: {quality_metrics['quality_score']:.2f}")
    
    return quality_metrics
