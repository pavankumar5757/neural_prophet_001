"""Data loading and preprocessing module for time series forecasting.

Handles:
- Loading data from Excel files
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
        """Load data from Excel file.
        
        Returns:
            pd.DataFrame: Loaded data with datetime index.
        """
        try:
            print(f"Loading data from: {self.config.data_file_path}")
            data = pd.read_excel(
                self.config.data_file_path,
                sheet_name=self.config.sheet_name
            )
            
            # Convert date column to datetime
            if self.config.date_column in data.columns:
                data[self.config.date_column] = pd.to_datetime(
                    data[self.config.date_column]
                )
                data = data.sort_values(by=self.config.date_column).reset_index(drop=True)
            
            self.data = data
            print(f"Data loaded successfully. Shape: {data.shape}")
            print(f"Date range: {data[self.config.date_column].min()} to {data[self.config.date_column].max()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data integrity.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Tuple[bool, List[str]]: Validation result and list of issues.
        """
        issues = []
        
        # Check required columns
        required_cols = [self.config.date_column, self.config.target_column]
        for col in required_cols:
            if col not in data.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check for empty data
        if data.empty:
            issues.append("Data is empty")
        
        # Check for duplicates
        if data[self.config.date_column].duplicated().any():
            issues.append(f"Duplicate dates found: {data[self.config.date_column].duplicated().sum()}")
        
        # Check for NaN values
        if data[self.config.target_column].isna().any():
            nan_count = data[self.config.target_column].isna().sum()
            issues.append(f"Missing values in target column: {nan_count}")
        
        # Check for non-numeric target
        try:
            pd.to_numeric(data[self.config.target_column])
        except:
            issues.append("Target column contains non-numeric values")
        
        if issues:
            print("\nData Validation Issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Data validation passed successfully!")
        
        return len(issues) == 0, issues
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            data: DataFrame with potential missing values.
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        if data[self.config.target_column].isna().sum() == 0:
            print("No missing values found.")
            return data
        
        original_na = data[self.config.target_column].isna().sum()
        print(f"Handling {original_na} missing values using '{self.config.handle_missing}' method...")
        
        if self.config.handle_missing == "drop":
            data = data.dropna(subset=[self.config.target_column]).reset_index(drop=True)
        
        elif self.config.handle_missing == "forward_fill":
            data[self.config.target_column] = data[self.config.target_column].fillna(method='ffill')
            data[self.config.target_column] = data[self.config.target_column].fillna(method='bfill')
        
        elif self.config.handle_missing == "interpolate":
            data[self.config.target_column] = data[self.config.target_column].interpolate(
                method='linear', limit_direction='both'
            )
        
        remaining_na = data[self.config.target_column].isna().sum()
        print(f"Remaining missing values: {remaining_na}")
        
        return data
    
    def detect_outliers(self, data: pd.DataFrame, 
                       method: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Detect and handle outliers using specified method.
        
        Args:
            data: DataFrame to process.
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest').
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Cleaned data and outlier mask.
        """
        method = method or self.config.outlier_detection_method
        y = data[self.config.target_column].values
        
        if method == "iqr":
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config.iqr_multiplier * IQR
            upper_bound = Q3 + self.config.iqr_multiplier * IQR
            outliers = (y < lower_bound) | (y > upper_bound)
        
        elif method == "zscore":
            mean = np.mean(y)
            std = np.std(y)
            outliers = np.abs((y - mean) / std) > self.config.outlier_threshold
        
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outliers = iso_forest.fit_predict(y.reshape(-1, 1)) == -1
        
        else:
            outliers = np.zeros(len(y), dtype=bool)
        
        outlier_count = outliers.sum()
        if outlier_count > 0:
            print(f"Detected {outlier_count} outliers using {method} method.")
        
        return data, outliers
    
    def normalize_data(self, data: pd.DataFrame, 
                      fit_scaler: bool = True) -> pd.DataFrame:
        """Normalize data using configured method.
        
        Args:
            data: DataFrame to normalize.
            fit_scaler: Whether to fit a new scaler or use existing one.
            
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if self.config.normalization_method == "minmax":
            if fit_scaler or self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(data[[self.config.target_column]])
            
            data[self.config.target_column] = self.scaler.transform(
                data[[self.config.target_column]]
            )
        
        elif self.config.normalization_method == "standard":
            if fit_scaler or self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(data[[self.config.target_column]])
            
            data[self.config.target_column] = self.scaler.transform(
                data[[self.config.target_column]]
            )
        
        print(f"Data normalized using {self.config.normalization_method} method.")
        return data
    
    def inverse_normalize(self, normalized_values: np.ndarray) -> np.ndarray:
        """Inverse normalize predictions back to original scale.
        
        Args:
            normalized_values: Normalized values.
            
        Returns:
            np.ndarray: Original scale values.
        """
        if self.scaler is None:
            print("Warning: Scaler not fitted. Returning original values.")
            return normalized_values
        
        return self.scaler.inverse_transform(normalized_values.reshape(-1, 1)).flatten()
    
    def split_temporal_data(self, data: pd.DataFrame, 
                           train_size: float = None,
                           validation_size: float = None) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets maintaining temporal order.
        
        Args:
            data: DataFrame to split.
            train_size: Proportion of data for training.
            validation_size: Proportion of data for validation.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with 'train', 'validation', 'test' splits.
        """
        train_size = train_size or (1 - self.config.test_size - self.config.validation_size)
        validation_size = validation_size or self.config.validation_size
        
        n = len(data)
        train_end = int(n * train_size)
        val_end = train_end + int(n * validation_size)
        
        train = data.iloc[:train_end].reset_index(drop=True)
        validation = data.iloc[train_end:val_end].reset_index(drop=True)
        test = data.iloc[val_end:].reset_index(drop=True)
        
        print(f"\nTemporal Data Split:")
        print(f"  Train: {len(train)} samples ({100*train_size:.1f}%)")
        print(f"  Validation: {len(validation)} samples ({100*validation_size:.1f}%)")
        print(f"  Test: {len(test)} samples ({100*self.config.test_size:.1f}%)")
        
        return {
            'train': train,
            'validation': validation,
            'test': test
        }
    
    def preprocess_complete(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Execute complete preprocessing pipeline.
        
        Args:
            data: DataFrame to process. If None, loads from file.
            
        Returns:
            pd.DataFrame: Fully preprocessed DataFrame.
        """
        if data is None:
            data = self.load_data()
        
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)
        
        # Validate
        valid, issues = self.validate_data(data)
        if not valid:
            raise ValueError(f"Data validation failed: {issues}")
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
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
