"""Feature engineering module for time series forecasting.

Implements advanced feature engineering techniques:
- Lag features (7-day, 30-day, 365-day)
- Rolling statistics (mean, std, min, max)
- Temporal features (day of week, month, etc.)
- Holiday indicators
- Seasonality components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from config import FeatureEngineeringConfig


class FeatureEngineer:
    """Generate advanced features for time series prediction."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        """Initialize FeatureEngineer with configuration.
        
        Args:
            config: FeatureEngineeringConfig object with feature settings.
        """
        self.config = config
        self.holidays_cache = {}
    
    def add_lag_features(self, data: pd.DataFrame, 
                        target_col: str,
                        lag_periods: Optional[List[int]] = None) -> pd.DataFrame:
        """Add lag features to the dataset.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            lag_periods: List of lag periods to create.
            
        Returns:
            pd.DataFrame: DataFrame with added lag features.
        """
        lag_periods = lag_periods or self.config.lag_values
        
        print(f"Adding lag features for periods: {lag_periods}")
        
        for lag in lag_periods:
            data[f'lag_{lag}'] = data[target_col].shift(lag)
        
        return data
    
    def add_rolling_stats(self, data: pd.DataFrame,
                         target_col: str,
                         window_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """Add rolling statistics as features.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            window_sizes: List of window sizes for rolling calculations.
            
        Returns:
            pd.DataFrame: DataFrame with added rolling features.
        """
        window_sizes = window_sizes or self.config.rolling_windows
        
        print(f"Adding rolling statistics for windows: {window_sizes}")
        
        for window in window_sizes:
            # Mean
            data[f'rolling_mean_{window}'] = data[target_col].rolling(window=window).mean()
            
            # Standard Deviation
            data[f'rolling_std_{window}'] = data[target_col].rolling(window=window).std()
            
            # Min and Max
            data[f'rolling_min_{window}'] = data[target_col].rolling(window=window).min()
            data[f'rolling_max_{window}'] = data[target_col].rolling(window=window).max()
            
            # Range (max - min)
            data[f'rolling_range_{window}'] = (
                data[f'rolling_max_{window}'] - data[f'rolling_min_{window}']
            )
        
        return data
    
    def add_temporal_features(self, data: pd.DataFrame,
                             date_col: str) -> pd.DataFrame:
        """Add temporal features based on date.
        
        Args:
            data: Input DataFrame.
            date_col: Date column name.
            
        Returns:
            pd.DataFrame: DataFrame with added temporal features.
        """
        print("Adding temporal features...")
        
        # Always add basic temporal features
        data['day_of_week'] = data[date_col].dt.dayofweek
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        data['month'] = data[date_col].dt.month
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        data['quarter'] = data[date_col].dt.quarter
        
        data['day_of_year'] = data[date_col].dt.dayofyear
        data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
        
        data['week_of_year'] = data[date_col].dt.isocalendar().week
        
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data
    
    def get_holidays(self, year: int) -> List[datetime]:
        """Get list of holidays for a given year.
        
        Args:
            year: Year to get holidays for.
            
        Returns:
            List[datetime]: List of holiday dates.
        """
        if year in self.holidays_cache:
            return self.holidays_cache[year]
        
        try:
            import holidays
            country_holidays = holidays.country_holidays(self.config.country_holidays)
            year_holidays = [date for date, name in sorted(country_holidays.items()) 
                            if date.year == year]
            self.holidays_cache[year] = year_holidays
            return year_holidays
        except:
            print(f"Warning: Could not load holidays for {self.config.country_holidays}")
            return []
    
    def add_holidays(self, data: pd.DataFrame,
                    date_col: str) -> pd.DataFrame:
        """Add holiday indicators as features.
        
        Args:
            data: Input DataFrame.
            date_col: Date column name.
            
        Returns:
            pd.DataFrame: DataFrame with added holiday features.
        """
        print(f"Adding holiday features for country: {self.config.country_holidays}")
        
        data['is_holiday'] = 0
        
        for year in data[date_col].dt.year.unique():
            holidays_list = self.get_holidays(year)
            
            # Mark holidays
            for holiday in holidays_list:
                mask = data[date_col].dt.date == holiday.date()
                data.loc[mask, 'is_holiday'] = 1
        
        return data
    
    def add_seasonality_features(self, data: pd.DataFrame,
                                target_col: str) -> pd.DataFrame:
        """Add seasonality components as features.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            
        Returns:
            pd.DataFrame: DataFrame with added seasonality features.
        """
        print(f"Adding seasonality features")
        
        # Use common seasonal periods
        seasonal_periods = [7, 30, 365]  # Weekly, monthly, yearly
        for period in seasonal_periods:
            # Seasonal mean
            seasonal_mean = data.groupby(data.index % period)[target_col].mean()
            data[f'seasonal_mean_{period}'] = data.index.map(
                lambda x: seasonal_mean[x % period]
            )
            
            # Deseasonalized values
            data[f'deseasonalized_{period}'] = data[target_col] - data[f'seasonal_mean_{period}']
        
        return data
    
    def add_differencing_features(self, data: pd.DataFrame,
                                 target_col: str,
                                 differences: List[int] = None) -> pd.DataFrame:
        """Add differencing features for trend capture.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            differences: List of difference orders (e.g., [1, 7, 30]).
            
        Returns:
            pd.DataFrame: DataFrame with added differencing features.
        """
        if differences is None:
            differences = [1, 7, 30]
        
        print(f"Adding differencing features for orders: {differences}")
        
        for diff in differences:
            data[f'diff_{diff}'] = data[target_col].diff(diff)
        
        return data
    
    def add_pct_change_features(self, data: pd.DataFrame,
                               target_col: str,
                               periods: List[int] = None) -> pd.DataFrame:
        """Add percentage change features.
        
        Args:
            data: Input DataFrame.
            target_col: Target column name.
            periods: List of periods for pct_change calculation.
            
        Returns:
            pd.DataFrame: DataFrame with added pct_change features.
        """
        if periods is None:
            periods = [1, 7, 30]
        
        print(f"Adding percentage change features for periods: {periods}")
        
        for period in periods:
            data[f'pct_change_{period}'] = data[target_col].pct_change(period)
        
        return data
    
    def create_all_features(self, data: pd.DataFrame,
                           date_col: str,
                           target_col: str) -> pd.DataFrame:
        """Execute complete feature engineering pipeline.
        
        Args:
            data: Input DataFrame.
            date_col: Date column name.
            target_col: Target column name.
            
        Returns:
            pd.DataFrame: DataFrame with all engineered features.
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        original_cols = set(data.columns)
        
        # Add temporal features first (no dependency)
        data = self.add_temporal_features(data, date_col)
        
        # Add lag features
        if self.config.add_lag_features:
            data = self.add_lag_features(data, target_col)
        
        # Add rolling statistics
        if self.config.add_rolling_features:
            data = self.add_rolling_stats(data, target_col)
        
        # Add seasonality
        data = self.add_seasonality_features(data, target_col)
        
        # Add holidays
        if self.config.add_holidays:
            data = self.add_holidays(data, date_col)
        
        # Add differencing
        data = self.add_differencing_features(data, target_col)
        
        # Add percentage changes
        data = self.add_pct_change_features(data, target_col)
        
        # Fill NaN values created by feature engineering
        data = data.bfill().ffill()
        
        new_cols = set(data.columns) - original_cols
        print(f"\nCreated {len(new_cols)} new features:")
        print(f"  Original columns: {len(original_cols)}")
        print(f"  New features: {len(new_cols)}")
        print(f"  Total columns: {len(data.columns)}")
        
        print("\n" + "="*60 + "\n")
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names.
        
        Returns:
            List[str]: List of feature names that will be created.
        """
        features = [
            'day_of_week', 'day_of_week_sin', 'day_of_week_cos',
            'month', 'month_sin', 'month_cos',
            'quarter',
            'day_of_year', 'day_of_year_sin', 'day_of_year_cos',
            'week_of_year',
            'is_weekend'
        ]
        
        return features


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    engineer = FeatureEngineer(config.features)
    
    # Example
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=365),
        'Demand': np.random.randn(365).cumsum() + 100
    })
    
    df_engineered = engineer.create_all_features(df, 'Date', 'Demand')
    print(f"\nFinal shape: {df_engineered.shape}")
    print(f"\nFeatures created:")
    print(df_engineered.columns.tolist())
