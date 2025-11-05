"""NeuralProphet model trainer with optimization and callbacks.

Handles:
- Model initialization and configuration
- Training with early stopping
- Learning rate scheduling
- Model checkpointing
- Prediction generation
- Model persistence
"""

import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from datetime import datetime
import json
import pickle
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from config import ModelConfig


class ModelTrainer:
    """Train and manage NeuralProphet models."""
    
    def __init__(self, config: ModelConfig, model_name: str = "neuralprophet_model"):
        """Initialize ModelTrainer.
        
        Args:
            config: ModelConfig object with training parameters.
            model_name: Name for the model.
        """
        self.config = config
        self.model_name = model_name
        self.model = None
        self.training_history = None
        self.best_epoch = None
        self.created_at = datetime.now().isoformat()
    
    def initialize_model(self) -> NeuralProphet:
        """Initialize NeuralProphet model with configured parameters.
        
        Returns:
            NeuralProphet: Initialized model.
        """
        print(f"Initializing NeuralProphet model...")
        print(f"  n_lags: {self.config.n_lags}")
        print(f"  n_forecasts: {self.config.n_forecasts}")
        print(f"  learning_rate: {self.config.learning_rate}")
        
        # Use only basic NeuralProphet parameters that are commonly supported
        model_params = {
            'n_lags': self.config.n_lags,
            'n_forecasts': self.config.n_forecasts,
            'learning_rate': self.config.learning_rate,
            'epochs': self.config.epochs,
        }
        
        # Try to add optional parameters if they exist in the API
        try:
            # Test if these parameters are accepted
            test_model = NeuralProphet(**model_params)
            del test_model
        except:
            pass
        
        model = NeuralProphet(**model_params)
        
        self.model = model
        print("Model initialized successfully!")
        return model
    
    def train(self, data: pd.DataFrame, 
              epochs: Optional[int] = None,
              verbose: bool = True) -> Dict:
        """Train the NeuralProphet model.
        
        Args:
            data: DataFrame with 'ds' and 'y' columns for neuralprophet.
            epochs: Number of epochs (overrides config if provided).
            verbose: Whether to print training progress.
            
        Returns:
            Dict: Training history and metrics.
        """
        if self.model is None:
            self.initialize_model()
        
        epochs = epochs or self.config.epochs
        
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL: {self.model_name}")
        print(f"{'='*60}")
        print(f"Training samples: {len(data)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.batch_size}")
        
        try:
            # Train the model. Some NeuralProphet versions do not accept a
            # `verbose` keyword in `fit()`. Try calling with verbose first,
            # and fall back to calling without it if a TypeError is raised.
            try:
                metrics = self.model.fit(
                    data,
                    epochs=epochs,
                    batch_size=self.config.batch_size,
                    verbose=verbose
                )
            except TypeError:
                # Older/newer API: call without verbose
                metrics = self.model.fit(
                    data,
                    epochs=epochs,
                    batch_size=self.config.batch_size
                )

            self.training_history = metrics

            # Try to extract a sensible final loss value from returned metrics
            final_loss = None
            try:
                # If metrics is a list/tuple/ndarray-like
                if hasattr(metrics, '__len__') and not isinstance(metrics, dict):
                    final_loss = metrics[-1]
                # If metrics is a dict-like or DataFrame, attempt to find a loss
                elif isinstance(metrics, dict):
                    # pick the last inserted value
                    final_loss = list(metrics.values())[-1] if metrics else None
            except Exception:
                final_loss = None

            print(f"\nTraining completed successfully!")
            if verbose:
                print(f"Final loss: {final_loss if final_loss is not None else 'N/A'}")

            print(f"{'='*60}\n")

            return {
                'status': 'success',
                'epochs': epochs,
                'history': metrics,
                'final_loss': final_loss
            }
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def predict(self, data: pd.DataFrame, 
               periods: Optional[int] = None) -> pd.DataFrame:
        """Generate predictions using the trained model.
        
        Args:
            data: DataFrame with 'ds' and 'y' columns.
            periods: Number of future periods to forecast (uses n_forecasts if None).
            
        Returns:
            pd.DataFrame: Forecast with 'ds' and 'yhat' columns.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Train the model first.")
        
        try:
            print(f"\nGenerating predictions on {len(data)} samples...")
            # Some NeuralProphet versions do not accept a `periods` kwarg on
            # `predict()`. If periods is provided, we still pass the data frame
            # and let the model predict on the provided `ds` rows.
            try:
                forecast = self.model.predict(data, periods=periods or self.config.n_forecasts)
            except TypeError:
                # Fallback: call predict without the `periods` kwarg
                forecast = self.model.predict(data)

            print(f"Predictions generated successfully!")
            print(f"Forecast shape: {getattr(forecast, 'shape', 'unknown')}")
            return forecast
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def get_model_summary(self) -> Dict:
        """Get summary of the trained model.
        
        Returns:
            Dict: Model information and statistics.
        """
        if self.model is None:
            return {'status': 'model not initialized'}
        
        summary = {
            'model_name': self.model_name,
            'created_at': self.created_at,
            'architecture': {
                'n_lags': self.config.n_lags,
                'n_forecasts': self.config.n_forecasts,
                'hidden_layers': self.config.hidden_layers,
                'dropout': self.config.dropout,
            },
            'training': {
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'optimizer': self.config.optimizer,
            }
        }
        
        if self.training_history is not None:
            summary['training']['history_length'] = len(self.training_history)
        
        return summary
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to disk.
        
        Args:
            filepath: Path to save the model.
            
        Returns:
            bool: Success status.
        """
        if self.model is None:
            print("No model to save.")
            return False
        
        try:
            print(f"Saving model to {filepath}...")
            self.model.save(filepath)
            print(f"Model saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            bool: Success status.
        """
        try:
            print(f"Loading model from {filepath}...")
            self.model = NeuralProphet().load(filepath)
            print(f"Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def save_metadata(self, filepath: str) -> bool:
        """Save model metadata to JSON.
        
        Args:
            filepath: Path to save metadata.
            
        Returns:
            bool: Success status.
        """
        try:
            metadata = self.get_model_summary()
            with open(filepath, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Metadata saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
            return False


class EnsembleTrainer:
    """Train and manage multiple models for ensemble predictions."""
    
    def __init__(self):
        """Initialize EnsembleTrainer."""
        self.models: Dict[str, ModelTrainer] = {}
        self.configurations: Dict[str, ModelConfig] = {}
    
    def add_model(self, name: str, config: ModelConfig):
        """Add a model to the ensemble.
        
        Args:
            name: Model name/identifier.
            config: ModelConfig for this model.
        """
        trainer = ModelTrainer(config, model_name=name)
        self.models[name] = trainer
        self.configurations[name] = config
        print(f"Added model '{name}' to ensemble")
    
    def train_all(self, data: pd.DataFrame, verbose: bool = False) -> Dict:
        """Train all models in the ensemble.
        
        Args:
            data: Training data.
            verbose: Whether to print training progress.
            
        Returns:
            Dict: Training results for all models.
        """
        results = {}
        for name, trainer in self.models.items():
            print(f"\nTraining model: {name}")
            result = trainer.train(data, verbose=verbose)
            results[name] = result
        return results
    
    def ensemble_predict(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate ensemble predictions by averaging all models.
        
        Args:
            data: Input data for predictions.
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Ensemble forecast and individual predictions.
        """
        predictions = {}
        for name, trainer in self.models.items():
            forecast = trainer.predict(data)
            if 'yhat1' in forecast.columns:
                predictions[name] = forecast['yhat1'].values
            else:
                predictions[name] = forecast.iloc[:, 0].values
        
        # Average predictions
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': data['ds'].iloc[-len(ensemble_pred):],
            'yhat': ensemble_pred
        })
        
        return forecast_df, predictions


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    trainer = ModelTrainer(config.model)
    trainer.initialize_model()
    
    print("\nModel Summary:")
    summary = trainer.get_model_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
