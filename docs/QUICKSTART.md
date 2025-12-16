# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Train with Synthetic Data (Default)

```bash
python train_pipeline.py
```

This will:
- Generate synthetic data with all feature types
- Train XGBoost, LSTM, and BERT models (if configured)
- Generate evaluation metrics and visualizations
- Save models and results

### 2. Train Only XGBoost

```bash
python train_pipeline.py --model xgboost
```

### 3. Use Your Own Data

```bash
python train_pipeline.py --data path/to/your/data.csv --model xgboost
```

## Expected Output

After running, you'll find:

- **Models**: `models/xgboost_model.pkl`, `models/best_lstm.pth`, `models/best_bert.pth`
- **Results**: 
  - `results/xgboost_predictions.png` - Prediction vs actual plots
  - `results/xgboost_residuals.png` - Residual analysis
  - `results/xgboost_feature_importance.png` - Feature importance
  - `results/xgboost_shap_*.png` - SHAP explanations
  - `results/xgboost_lime_*.png` - LIME explanations

## Data Format

Your CSV should have:
- **Target**: `energy_efficiency` (0-1) or `energy_consumption` (kWh)
- **Features**: Any combination of:
  - IoT: `temperature`, `humidity`, `air_quality`, `equipment_status`
  - Design: `orientation`, `material`, `roof_type`, `insulation`
  - Survey: `green_perception`, `environmental_awareness`, `perceived_risk`
  - Climate: `solar_radiation`, `wind_speed`, `temperature_profile`
  - Operational: `metro_logistics`, `policy_compliance`

## Configuration

Edit `config.json` to customize:
- Model hyperparameters
- Preprocessing options
- Evaluation settings

## Troubleshooting

**Memory errors**: Reduce batch size in config.json
**Missing packages**: `pip install --upgrade -r requirements.txt`
**BERT errors**: BERT requires significant memory, use smaller batch sizes

