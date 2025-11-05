"""Data loading and preprocessing module for time series forecasting.
Handles:
- Loading data from CSV files
- Data validation and cleaning
- Missing value handling
- Outlier detection and treatment
- Normalization/standardization
- Temporal data splitting
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')
from config import DataConfig

class DataLoader:
    """Load and preprocess time series data."""
    
    def __init__(self, config: DataConfig):
        """Initialize DataLoader with configuration.
        
        Args:
            config: DataConfig object with data settings.
        """
        self.config = config
        self.data = None
        self.scaler = None
        self.target_scaler = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.
        
        Returns:
            pd.DataFrame: Raw data loaded from CSV.
        """
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load CSV file - skip first 5 header rows
        data = pd.read_csv(
            self.config.data_file_path,
            skiprows=5,
            parse_dates=[self.config.date_column]
        )
        
        print(f"[OK] Data loaded from: {self.config.data_file_path}")
        print(f"  - Rows: {len(data)}")
        print(f"  - Columns: {list(data.columns)}")
        print(f"  - Date range: {data[self.config.date_column].min()} to {data[self.config.date_column].max()}")
        
        # Validate required columns
        if self.config.date_column not in data.columns:
            raise ValueError(f"Date column '{self.config.date_column}' not found in data")
        if self.config.target_column not in data.columns:
            raise ValueError(f"Target column '{self.config.target_column}' not found in data")
            
        return data
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data for required structure and content.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            pd.DataFrame: Validated data.
        """
        print("\nValidating data...")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[self.config.date_column]):
            data[self.config.date_column] = pd.to_datetime(data[self.config.date_column])
        
        # Sort by date
        data = data.sort_values(self.config.date_column).reset_index(drop=True)
        
        # Check for duplicates
        duplicates = data[self.config.date_column].duplicated().sum()
        if duplicates > 0:
            print(f"  [WARN] Warning: {duplicates} duplicate dates found, keeping first occurrence")
            data = data.drop_duplicates(subset=[self.config.date_column], keep='first')
        
        print("[OK] Data validated successfully")
        return data
    
    def handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            pd.DataFrame: Data with missing values handled.
        """
        print("\nHandling missing values...")
        
        missing_count = data[self.config.target_column].isna().sum()
        print(f"  - Missing values in target: {missing_count}")
        
        if missing_count == 0:
            print("[OK] No missing values found")
            return data
        
        if self.config.handle_missing == 'interpolate':
            data[self.config.target_column] = data[self.config.target_column].interpolate(method='linear')
            print("[OK] Missing values interpolated")
        elif self.config.handle_missing == 'forward_fill':
            data[self.config.target_column] = data[self.config.target_column].ffill()
            print("[OK] Missing values forward-filled")
        elif self.config.handle_missing == 'drop':
            data = data.dropna(subset=[self.config.target_column])
            print(f"[OK] Dropped {missing_count} rows with missing values")
        else:
            print("[WARN] Unknown method, keeping missing values")
        
        return data
    
    def detect_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Detect outliers using z-score method.
        
        Args:
            data: Input DataFrame.
            
        Returns:
            Tuple of (data, outlier_flags)
        """
        print("\nDetecting outliers...")
        
        if not self.config.remove_outliers:
            print("[OK] Outlier detection skipped (remove_outliers=False)")
            return data, pd.Series([False] * len(data))
        
        # Calculate z-scores
        mean = data[self.config.target_column].mean()
        std = data[self.config.target_column].std()
        z_scores = np.abs((data[self.config.target_column] - mean) / std)
        
        outliers = z_scores > self.config.outlier_threshold
        outlier_count = outliers.sum()
        
        print(f"  - Outliers detected: {outlier_count}")
        
        if outlier_count > 0 and self.config.remove_outliers:
            print(f"[OK] Outliers flagged (threshold: {self.config.outlier_threshold} std)")
        
        return data, outliers
    
    def normalize_data(self, data: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize or standardize the target variable.
        
        Args:
            data: Input DataFrame.
            fit_scaler: Whether to fit the scaler (True for training data).
            
        Returns:
            pd.DataFrame: Data with normalized target.
        """
        print("\nNormalizing data...")
        
        if not self.config.normalize:
            print("[OK] Normalization skipped (normalize=False)")
            return data
        
        if fit_scaler:
            self.target_scaler = MinMaxScaler(feature_range=(0, 1))
            data[self.config.target_column] = self.target_scaler.fit_transform(
                data[[self.config.target_column]]
            )
            print("[OK] Data normalized (fitted scaler)")
        else:
            if self.target_scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            data[self.config.target_column] = self.target_scaler.transform(
                data[[self.config.target_column]]
            )
            print("[OK] Data normalized (using existing scaler)")
        
        return data
    
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform normalized values back to original scale.
        
        Args:
            values: Normalized values.
            
        Returns:
            np.ndarray: Original scale values.
        """
        if self.target_scaler is None:
            return values
        return self.target_scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    
    def split_temporal_data(
        self, 
        data: pd.DataFrame, 
        train_ratio: float = 0.7, 
        val_ratio: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """Split data temporally (chronologically) into train/val/test sets.
        
        Args:
            data: Input DataFrame.
            train_ratio: Proportion of data for training.
            val_ratio: Proportion of data for validation.
            
        Returns:
            Dictionary with 'train', 'val', and 'test' DataFrames.
        """
        print("\nSplitting data temporally...")
        
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data.iloc[:train_size].copy()
        val_data = data.iloc[train_size:train_size + val_size].copy()
        test_data = data.iloc[train_size + val_size:].copy()
        
        print(f"  - Train: {len(train_data)} samples ({train_ratio*100:.1f}%)")
        print(f"  - Val:   {len(val_data)} samples ({val_ratio*100:.1f}%)")
        print(f"  - Test:  {len(test_data)} samples ({(1-train_ratio-val_ratio)*100:.1f}%)")
        print("[OK] Data split completed")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def preprocess_complete(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Complete preprocessing pipeline.
        
        Returns:
            pd.DataFrame: Fully preprocessed data.
        """
        print("\n" + "="*60)
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load if data not provided
        if data is None:
            data = self.load_data()
        
        # Validate
        data = self.validate_data(data)
        
        # Handle missing values
        data = self.handle_missing(data)
        
        # Detect outliers (flag but don't remove)
        data, outliers = self.detect_outliers(data)
        
        # Normalize
        data = self.normalize_data(data, fit_scaler=True)
        
        print("\nPreprocessing completed successfully!")
        print("="*60 + "\n")
        
        self.data = data
        return data

if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    loader = DataLoader(config.data)
    
    data = loader.preprocess_complete()
    splits = loader.split_temporal_data(data)
    
    print("\nData splits created successfully!")
    for name, split in splits.items():
        print(f"{name}: {len(split)} samples")
