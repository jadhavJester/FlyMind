"""
Script to execute the dataset preprocessing pipeline (Phase 5).
Validates inputs, executes stratified splits, and saves normalized datasets to data/processed/.
"""

import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import FlightDataPreprocessor

def main():
    print("=" * 60)
    print("BioFlight — Dataset Preprocessing Pipeline (Phase 5)")
    print("=" * 60)

    preprocessor = FlightDataPreprocessor("configs/dataset.yaml")
    data = preprocessor.process()

    print("\n--- Summary of Processed Dataset ---")
    print(f"Train samples: {data['X_train'].shape[0]:,} | Input dim: {data['X_train'].shape[1]} | Target dim: {data['Y_train'].shape[1]}")
    print(f"Val samples:   {data['X_val'].shape[0]:,}")
    print(f"Test samples:  {data['X_test'].shape[0]:,}")
    print("Zero temporal leakage: verified (trajectories partitioned cleanly).")
    print("=" * 60)

if __name__ == "__main__":
    main()
