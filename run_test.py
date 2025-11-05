import sys
import os
sys.stdout = sys.stderr  # Force unbuffered output

print("Starting test run...")
print("Importing main...")

try:
    from main import main
    print("Starting main()...")
    results = main()
    print("Main completed!")
    print(f"Results status: {results.get('status', 'unknown')}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

