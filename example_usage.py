"""
Example usage of the Green Building Energy Efficiency Prediction Pipeline
"""

from train_pipeline import EnergyEfficiencyPipeline
import pandas as pd

def main():
    """Example usage"""
    
    # Initialize pipeline with default config
    pipeline = EnergyEfficiencyPipeline(config_path='config.json')
    
    # Option 1: Use synthetic data (default)
    print("Running with synthetic data...")
    pipeline.run()
    
    # Option 2: Use your own data
    # pipeline.run(data_path='data/your_data.csv')
    
    # Option 3: Train specific model
    # pipeline.config['model_type'] = 'xgboost'
    # pipeline.run()
    
    # Access trained models
    if 'xgboost' in pipeline.models:
        xgboost_model = pipeline.models['xgboost']
        print("\nXGBoost model trained successfully!")
        
        # Make predictions on new data
        # X_new = ...  # Your new data
        # predictions = xgboost_model.predict(X_new)
    
    if 'lstm' in pipeline.models:
        lstm_model = pipeline.models['lstm']['model']
        print("LSTM model trained successfully!")
    
    if 'bert' in pipeline.models:
        bert_model = pipeline.models['bert']['model']
        print("BERT model trained successfully!")

if __name__ == '__main__':
    main()

