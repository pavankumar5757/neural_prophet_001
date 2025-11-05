from neuralprophet import NeuralProphet
import pandas as pd

# Test minimal initialization
print("Testing minimal NeuralProphet...")
try:
    model = NeuralProphet(n_lags=10, n_forecasts=5)
    print("✓ Minimal model works!")
except Exception as e:
    print(f"✗ Error: {e}")

# Test with more parameters
print("\nTesting with more parameters...")
try:
    model = NeuralProphet(
        n_lags=10,
        n_forecasts=5,
        hidden_size=128,
        dropout=0.1,
        learning_rate=0.001,
        epochs=10
    )
    print("✓ Extended model works!")
except Exception as e:
    print(f"✗ Error: {e}")

