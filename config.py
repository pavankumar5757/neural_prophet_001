"""Configuration management for NeuralProphet time series forecasting.

This module centralizes all configuration parameters for the neural_prophet_001 project,
including model architecture, training parameters, data settings, and cross-validation options.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os


@dataclass
class ModelConfig:
    """NeuralProphet model architecture and training parameters."""
    
    # Model Architecture
    n_lags: int = 180  # Number of past time steps to use as input
    n_forecasts: int = 30  # Number of future time steps to predict
    hidden_layers: List[int] = None  # Hidden layer sizes
    num_hidden_layers: int = 3
    hidden_size: int = 256
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Training Parameters
    epochs: int = 200
    batch_size: int = 32
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.001
    gradient_clip_norm: float = 1.0
    
    # Optimization
    optimizer: str = "Adam"
    use_learning_rate_scheduler: bool = True
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [self.hidden_size // (2**i) for i in range(self.num_hidden_layers)]


@dataclass
class DataConfig:
    """Data processing and pipeline parameters."""
    
    # File Paths
    data_file_path: str = "data/Trend DC SG AG (1).xlsx"
    output_dir: str = "outputs"
    models_dir: str = "models"
    plots_dir: str = "plots"
    
    # Data Settings
    sheet_name: str = "Sheet1"
    date_column: str = "Date"
    target_column: str = "Demand"
    
    # Data Processing
    test_size: float = 0.15
    validation_size: float = 0.1
    normalization_method: str = "minmax"  # 'minmax' or 'standard'
    handle_missing: str = "interpolate"  # 'drop', 'forward_fill', 'interpolate'
    
    # Outlier Detection
    outlier_detection_method: str = "iqr"  # 'iqr', 'zscore', 'isolation_forest'
    outlier_threshold: float = 3.0  # For z-score
    iqr_multiplier: float = 1.5
    
    # Feature Settings
    use_lag_features: bool = True
    use_rolling_stats: bool = True
    use_temporal_features: bool = True
    use_holidays: bool = True
    use_seasonality: bool = True
    
    # Lag Configuration
    lag_periods: List[int] = None
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [7, 30, 365]  # Weekly, monthly, yearly lags
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.output_dir, self.models_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)


@dataclass
class CrossValidationConfig:
    """Cross-validation strategy parameters."""
    
    # CV Strategy
    strategy: str = "rolling_origin"  # 'rolling_origin' or 'time_series_split'
    n_splits: int = 3
    gap: int = 0  # Gap between train and test sets
    
    # Split Strategy (for rolling origin)
    train_years: List[List[int]] = None  # [[2019, 2020], [2019, 2021], [2019, 2022]]
    test_years: List[List[int]] = None   # [[2021], [2022], [2023]]
    
    # Metrics Configuration
    primary_metric: str = "mape"  # Primary metric for model selection
    calculate_confidence_intervals: bool = True
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.train_years is None:
            self.train_years = []
        if self.test_years is None:
            self.test_years = []


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering parameters."""
    
    # Lag Features
    rolling_window_sizes: List[int] = None
    
    # Temporal Features
    include_day_of_week: bool = True
    include_month: bool = True
    include_quarter: bool = True
    include_day_of_year: bool = True
    include_week_of_year: bool = True
    include_is_weekend: bool = True
    
    # Holiday Configuration
    holiday_country: str = "US"
    include_pre_holiday: bool = True
    include_post_holiday: bool = True
    holiday_lag_days: int = 1
    
    # Seasonality
    seasonal_periods: List[int] = None  # [7, 30, 365]
    
    def __post_init__(self):
        if self.rolling_window_sizes is None:
            self.rolling_window_sizes = [7, 30]
        if self.seasonal_periods is None:
            self.seasonal_periods = [7, 30, 365]


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""
    
    # Metrics to Calculate
    calculate_mae: bool = True
    calculate_rmse: bool = True
    calculate_mape: bool = True
    calculate_r2: bool = True
    calculate_smape: bool = True
    calculate_median_ae: bool = True
    
    # Error Analysis
    analyze_directional_accuracy: bool = True
    analyze_forecast_bias: bool = True
    analyze_residuals: bool = True
    
    # Comparison Baseline
    use_naive_baseline: bool = True
    use_seasonal_naive: bool = True
    
    # Report Generation
    generate_detailed_report: bool = True
    include_error_distribution: bool = True
    include_residual_diagnostics: bool = True


@dataclass
class VisualizationConfig:
    """Visualization parameters."""
    
    # Plot Settings
    figure_size: Tuple[int, int] = (14, 7)
    dpi: int = 300
    style: str = "seaborn-v0_8-darkgrid"
    
    # Plot Types
    plot_forecast: bool = True
    plot_residuals: bool = True
    plot_components: bool = True
    plot_metrics: bool = True
    plot_cv_results: bool = True
    plot_feature_importance: bool = True
    
    # Forecast Periods for Visualization
    forecast_periods: List[int] = None
    
    def __post_init__(self):
        if self.forecast_periods is None:
            self.forecast_periods = [30, 90, 180, 365]


class ProjectConfig:
    """Master configuration class combining all sub-configurations."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.cv = CrossValidationConfig()
        self.features = FeatureEngineeringConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
    
    def initialize(self):
        """Initialize all configuration settings."""
        self.data.ensure_directories()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'cv': self.cv.__dict__,
            'features': self.features.__dict__,
            'evaluation': self.evaluation.__dict__,
            'visualization': self.visualization.__dict__,
        }
    
    def display(self):
        """Display current configuration."""
        print("="*80)
        print("PROJECT CONFIGURATION")
        print("="*80)
        
        print("\nMODEL CONFIG:")
        for key, value in self.model.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nDATA CONFIG:")
        for key, value in self.data.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nCROSS-VALIDATION CONFIG:")
        for key, value in self.cv.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nFEATURE ENGINEERING CONFIG:")
        for key, value in self.features.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nEVALUATION CONFIG:")
        for key, value in self.evaluation.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\nVISUALIZATION CONFIG:")
        for key, value in self.visualization.__dict__.items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*80)


def get_config() -> ProjectConfig:
    """Factory function to get project configuration.
    
    Returns:
        ProjectConfig: Initialized project configuration object.
    """
    config = ProjectConfig()
    config.initialize()
    return config


if __name__ == "__main__":
    config = get_config()
    config.display()
