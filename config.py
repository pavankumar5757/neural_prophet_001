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
            self.hidden_layers = [self.hidden_size] * self.num_hidden_layers

@dataclass
class DataConfig:
    """Data loading and preprocessing parameters."""
    
    # File paths
    data_file_path: str = "data/data.csv"
    output_dir: str = "outputs/"
    plots_dir: str = "plots/"
    models_dir: str = "models/"
    
    # CSV parameters (removed sheet_name)
    date_column: str = "ds"  # Column name for dates
    target_column: str = "y"  # Column name for target variable
    
    # Data preprocessing
    handle_missing: str = "interpolate"  # Options: 'interpolate', 'forward_fill', 'drop'
    normalize: bool = True
    log_transform: bool = False
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

@dataclass
class CrossValidationConfig:
    """Rolling-origin cross-validation parameters."""
    
    # Rolling window parameters
    initial_train_size: int = 365  # Days in initial training window
    test_size: int = 30  # Days to forecast in each fold
    step_size: int = 30  # Days to roll forward between folds
    max_folds: Optional[int] = None  # Maximum number of folds (None = all possible)
    
    # Validation strategy
    use_gap: bool = False  # Whether to add gap between train and test
    gap_size: int = 0  # Days of gap if use_gap=True
    expanding_window: bool = True  # True = expanding window, False = sliding window
    
    # Ensemble settings
    use_ensemble: bool = False
    ensemble_method: str = "mean"  # Options: 'mean', 'median', 'weighted'

@dataclass
class FeatureEngineeringConfig:
    """Feature engineering parameters."""
    
    # Temporal features
    add_seasonality: bool = True
    add_holidays: bool = True
    country_holidays: str = "US"  # Country code for holidays
    
    # Lag features
    add_lag_features: bool = True
    lag_values: List[int] = None
    
    # Rolling features
    add_rolling_features: bool = True
    rolling_windows: List[int] = None
    
    # Fourier features
    add_fourier_features: bool = True
    yearly_seasonality: int = 10
    weekly_seasonality: int = 3
    daily_seasonality: int = 4
    
    def __post_init__(self):
        if self.lag_values is None:
            self.lag_values = [7, 14, 30, 90]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 30]

@dataclass
class EvaluationConfig:
    """Model evaluation parameters."""
    
    # Metrics to compute
    compute_mae: bool = True
    compute_rmse: bool = True
    compute_mape: bool = True
    compute_r2: bool = True
    compute_mase: bool = True
    
    # Confidence intervals
    confidence_level: float = 0.95
    compute_prediction_intervals: bool = True
    
    # Error analysis
    analyze_residuals: bool = True
    test_stationarity: bool = True
    test_autocorrelation: bool = True

@dataclass
class VisualizationConfig:
    """Plotting and visualization parameters."""
    
    # Plot settings
    figure_size: Tuple[int, int] = (15, 6)
    dpi: int = 100
    style: str = "seaborn-v0_8-darkgrid"
    
    # Components to plot
    plot_components: bool = True
    plot_parameters: bool = False
    plot_trend: bool = True
    plot_seasonality: bool = True
    
    # Forecast plots
    show_confidence_intervals: bool = True
    highlight_forecast: bool = True
    
    # Save settings
    save_plots: bool = True
    plot_format: str = "png"  # Options: 'png', 'pdf', 'svg'

class ProjectConfig:
    """Main project configuration container."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.cv = CrossValidationConfig()
        self.features = FeatureEngineeringConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()
        
    def initialize(self):
        """Initialize all sub-configurations."""
        # Trigger __post_init__ for all configs
        self.model.__post_init__()
        self.data.__post_init__()
        self.features.__post_init__()
        
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
