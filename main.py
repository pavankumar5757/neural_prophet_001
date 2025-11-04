"""Main orchestration script for neural_prophet_001 improved project.

Executes complete pipeline:
- Data loading and preprocessing
- Feature engineering
- Rolling-origin cross-validation
- Model training
- Evaluation and comparison
- Result visualization
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from evaluation import Evaluator
from visualizer import Visualizer
from rolling_cv import RollingOriginCV
from utils import yearly_train_test_split, plot_actual_vs_predicted, plot_forecasts


class NeuralProphetPipeline:
    """Complete neural prophet forecasting pipeline."""
    
    def __init__(self, config=None):
        """Initialize pipeline with configuration.
        
        Args:
            config: ProjectConfig object (uses defaults if None).
        """
        self.config = config or get_config()
        self.data_loader = DataLoader(self.config.data)
        self.feature_engineer = FeatureEngineer(self.config.features)
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()
        
        self.data = None
        self.train_data = None
        self.test_data = None
        self.model_trainer = None
        self.results = {}
        
        print("\n" + "="*80)
        print("NEURAL PROPHET TIME SERIES FORECASTING - IMPROVED V2.0")
        print("="*80 + "\n")
        
        # Initialize directories
        self.config.data.ensure_directories()
    
    def execute_pipeline(self) -> Dict:
        """Execute complete forecasting pipeline.
        
        Returns:
            Dict: Pipeline results and metrics.
        """
        try:
            # Step 1: Load and preprocess data
            print("\n[STEP 1/5] Loading and preprocessing data...")
            self.data = self.load_and_preprocess()
            
            # Step 2: Feature engineering
            print("\n[STEP 2/5] Engineering features...")
            self.data = self.apply_feature_engineering()
            
            # Step 3: Train/test split
            print("\n[STEP 3/5] Splitting data...")
            self.train_data, self.test_data = self.split_data()
            
            # Step 4: Train model with cross-validation
            print("\n[STEP 4/5] Training model with cross-validation...")
            cv_results = self.train_with_cv()
            self.results['cv_results'] = cv_results
            
            # Step 5: Evaluate and visualize
            print("\n[STEP 5/5] Evaluating model and generating visualizations...")
            eval_results = self.evaluate_and_visualize()
            self.results['evaluation'] = eval_results
            
            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("="*80 + "\n")
            
            return self.results
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'failed', 'error': str(e)}
    
    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess data.
        
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        # Load
        data = self.data_loader.load_data()
        
        # Preprocess (validate, handle missing, normalize)
        data = self.data_loader.preprocess_complete(data)
        
        return data
    
    def apply_feature_engineering(self) -> pd.DataFrame:
        """Apply feature engineering to data.
        
        Returns:
            pd.DataFrame: Data with engineered features.
        """
        # Create all features
        data = self.feature_engineer.create_all_features(
            self.data.copy(),
            self.config.data.date_column,
            self.config.data.target_column
        )
        
        return data
    
    def split_data(self) -> tuple:
        """Split data into train and test sets.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test data.
        """
        splits = self.data_loader.split_temporal_data(self.data)
        return splits['train'], splits['test']
    
    def train_with_cv(self) -> Dict:
        """Train model with rolling-origin cross-validation.
        
        Returns:
            Dict: Cross-validation results.
        """
        cv = RollingOriginCV(self.config.cv, self.config.model, self.config.data)
        cv_results = cv.cross_validate(self.data)
        
        return cv_results
    
    def evaluate_and_visualize(self) -> Dict:
        """Evaluate model on test set and generate visualizations.
        
        Returns:
            Dict: Evaluation results.
        """
        # Prepare NeuralProphet data format
        train_np = self.train_data[[
            self.config.data.date_column, 
            self.config.data.target_column
        ]].copy()
        train_np.columns = ['ds', 'y']
        
        test_np = self.test_data[[
            self.config.data.date_column,
            self.config.data.target_column
        ]].copy()
        test_np.columns = ['ds', 'y']
        
        # Train final model
        self.model_trainer = ModelTrainer(self.config.model)
        self.model_trainer.initialize_model()
        train_result = self.model_trainer.train(train_np, verbose=False)
        
        # Predict on test set
        if train_result['status'] == 'success':
            forecast = self.model_trainer.predict(test_np)
            
            # Extract predictions
            actual = test_np['y'].values
            predicted = forecast['yhat1'].values[:len(actual)]
            
            # Calculate metrics
            metrics = self.evaluator.calculate_all_metrics(actual, predicted)
            
            # Generate visualizations
            self.generate_visualizations(actual, predicted, forecast, test_np)
            
            print("\nFinal Test Metrics:")
            self.evaluator.print_metrics_report(metrics)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'model_summary': self.model_trainer.get_model_summary()
            }
        else:
            return {'status': 'failed', 'error': train_result.get('error')}
    
    def generate_visualizations(self, actual: np.ndarray, 
                               predicted: np.ndarray,
                               forecast: pd.DataFrame,
                               test_data: pd.DataFrame) -> None:
        """Generate all visualizations.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            forecast: Full forecast dataframe.
            test_data: Test dataset.
        """
        output_dir = self.config.data.plots_dir
        
        # Actual vs Predicted
        self.visualizer.plot_actual_vs_predicted(
            actual, predicted,
            save_path=os.path.join(output_dir, "actual_vs_predicted.png")
        )
        
        # Residuals
        self.visualizer.plot_residuals(
            actual, predicted,
            save_path=os.path.join(output_dir, "residuals_analysis.png")
        )
        
        # Error distribution
        self.visualizer.plot_error_distribution(
            actual, predicted,
            save_path=os.path.join(output_dir, "error_distribution.png")
        )
        
        print(f"\nVisualizations saved to {output_dir}")
    
    def save_results(self, filepath: str = None) -> None:
        """Save results to file.
        
        Args:
            filepath: Path to save results JSON.
        """
        if filepath is None:
            filepath = os.path.join(
                self.config.data.output_dir,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        import json
        results_to_save = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'results': str(self.results)  # Convert to string for JSON serialization
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")


def main():
    """Main entry point."""
    # Initialize configuration
    config = get_config()
    
    # Display configuration
    print("\nProject Configuration:")
    print("-" * 80)
    print(f"Data file: {config.data.data_file_path}")
    print(f"Target column: {config.data.target_column}")
    print(f"Model: NeuralProphet")
    print(f"  - n_lags: {config.model.n_lags}")
    print(f"  - n_forecasts: {config.model.n_forecasts}")
    print(f"  - hidden_layers: {config.model.hidden_layers}")
    print(f"  - epochs: {config.model.epochs}")
    print(f"CV Strategy: {config.cv.strategy}")
    print(f"CV Splits: {config.cv.n_splits}")
    print("-" * 80 + "\n")
    
    # Execute pipeline
    pipeline = NeuralProphetPipeline(config)
    results = pipeline.execute_pipeline()
    
    # Save results
    pipeline.save_results()
    
    return results


if __name__ == "__main__":
    results = main()
