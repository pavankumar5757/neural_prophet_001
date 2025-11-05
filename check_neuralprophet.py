from neuralprophet import NeuralProphet
import inspect

sig = inspect.signature(NeuralProphet.__init__)
params = list(sig.parameters.keys())[1:]  # Skip 'self'
print("NeuralProphet parameters:")
for p in params:
    print(f"  - {p}")

# Try to create a model with minimal params
print("\nTrying minimal model...")
try:
    model = NeuralProphet(n_lags=10, n_forecasts=5)
    print("Success with minimal params!")
except Exception as e:
    print(f"Error: {e}")

