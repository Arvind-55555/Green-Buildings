"""
Synthetic Data Generator for Green Building Energy Efficiency
Generates realistic data for all feature categories
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic data for green building energy efficiency prediction
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    
    data = {}
    
    # IoT Sensor Data
    data['temperature'] = np.random.normal(22, 5, n_samples)  # Celsius
    data['humidity'] = np.random.uniform(30, 70, n_samples)  # Percentage
    data['air_quality'] = np.random.uniform(0, 100, n_samples)  # AQI
    data['equipment_status'] = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.7, 0.2])  # 0=off, 1=on, 2=maintenance
    
    # Building Design Parameters
    data['orientation'] = np.random.choice([0, 90, 180, 270], n_samples)  # Degrees
    data['material'] = np.random.choice(['concrete', 'steel', 'wood', 'composite'], n_samples)
    data['roof_type'] = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2])  # 0=flat, 1=pitched, 2=green
    data['insulation'] = np.random.uniform(0.1, 0.5, n_samples)  # R-value
    
    # Consumer Survey Data
    data['green_perception'] = np.random.uniform(1, 5, n_samples)  # 1-5 scale
    data['environmental_awareness'] = np.random.uniform(1, 5, n_samples)  # 1-5 scale
    data['perceived_risk'] = np.random.uniform(1, 5, n_samples)  # 1-5 scale
    
    # Climate Data
    data['solar_radiation'] = np.random.uniform(100, 800, n_samples)  # W/m²
    data['wind_speed'] = np.random.uniform(0, 15, n_samples)  # m/s
    data['temperature_profile'] = np.random.normal(20, 8, n_samples)  # Average temperature
    
    # Stakeholder and Operational Data
    data['metro_logistics'] = np.random.uniform(0, 1, n_samples)  # Efficiency score
    data['policy_compliance'] = np.random.uniform(0.5, 1.0, n_samples)  # Compliance rate
    
    # Timestamp (for temporal features)
    start_date = datetime(2020, 1, 1)
    timestamps = [start_date + timedelta(days=i, hours=random.randint(0, 23)) 
                 for i in range(n_samples)]
    data['timestamp'] = timestamps
    
    # Create target variable (Energy Efficiency)
    # Energy efficiency is influenced by multiple factors
    efficiency = (
        0.3 * (data['temperature'] / 25) +  # Temperature effect
        0.2 * (1 - data['humidity'] / 100) +  # Humidity effect (inverse)
        0.15 * (data['air_quality'] / 100) +  # Air quality effect
        0.1 * (data['solar_radiation'] / 800) +  # Solar gain
        0.1 * (data['insulation'] * 2) +  # Insulation effect
        0.05 * (data['green_perception'] / 5) +  # Perception effect
        0.05 * (data['policy_compliance']) +  # Compliance effect
        0.05 * (data['metro_logistics'])  # Logistics effect
    )
    
    # Add noise
    efficiency += np.random.normal(0, 0.1, n_samples)
    
    # Normalize to 0-1 range (energy efficiency score)
    efficiency = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min())
    
    # Convert to energy consumption (kWh) - inverse relationship
    # Higher efficiency = lower consumption
    data['energy_consumption'] = 1000 * (1 - efficiency) + np.random.normal(0, 50, n_samples)
    data['energy_efficiency'] = efficiency
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values (5% missing)
    missing_cols = ['temperature', 'humidity', 'air_quality', 'solar_radiation']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


def generate_temporal_data(n_samples=2000, seq_length=24):
    """
    Generate temporal/sequential data for LSTM model
    
    Args:
        n_samples: Total number of time steps
        seq_length: Length of sequences
    """
    np.random.seed(42)
    
    # Generate time series data
    time_steps = np.arange(n_samples)
    
    # Create features with temporal patterns
    temperature = 20 + 5 * np.sin(2 * np.pi * time_steps / 365) + np.random.normal(0, 1, n_samples)
    humidity = 50 + 10 * np.sin(2 * np.pi * time_steps / 365 + np.pi) + np.random.normal(0, 3, n_samples)
    solar_radiation = 400 + 200 * np.sin(2 * np.pi * time_steps / 365) + np.random.normal(0, 50, n_samples)
    wind_speed = 5 + 3 * np.sin(2 * np.pi * time_steps / 365) + np.random.normal(0, 1, n_samples)
    
    # Energy efficiency with temporal dependencies
    efficiency = (
        0.4 * (temperature / 25) +
        0.3 * (solar_radiation / 800) +
        0.2 * (1 - humidity / 100) +
        0.1 * (wind_speed / 15)
    ) + np.random.normal(0, 0.05, n_samples)
    
    efficiency = np.clip(efficiency, 0, 1)
    energy_consumption = 1000 * (1 - efficiency) + np.random.normal(0, 30, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'temperature': temperature,
        'humidity': humidity,
        'solar_radiation': solar_radiation,
        'wind_speed': wind_speed,
        'energy_consumption': energy_consumption,
        'energy_efficiency': efficiency
    })
    
    return df


def generate_text_data(n_samples=500):
    """
    Generate synthetic text data for BERT model (survey responses, descriptions)
    
    Args:
        n_samples: Number of text samples
    """
    np.random.seed(42)
    
    # Template responses
    green_perception_templates = [
        "This building demonstrates excellent environmental sustainability practices.",
        "The green building features are impressive and well-implemented.",
        "Moderate environmental consciousness in building design.",
        "Limited green building initiatives observed.",
        "Building lacks significant environmental considerations."
    ]
    
    awareness_templates = [
        "High awareness of energy efficiency and environmental impact.",
        "Good understanding of sustainable building practices.",
        "Some awareness of environmental issues.",
        "Limited knowledge about green building concepts.",
        "No significant environmental awareness demonstrated."
    ]
    
    risk_templates = [
        "Very low perceived risk associated with green building technologies.",
        "Low risk perception, confident in building systems.",
        "Moderate concerns about technology reliability.",
        "Some perceived risks with new building systems.",
        "High perceived risk and uncertainty about green technologies."
    ]
    
    texts = []
    green_scores = []
    awareness_scores = []
    risk_scores = []
    
    for i in range(n_samples):
        # Sample templates
        green_text = np.random.choice(green_perception_templates)
        awareness_text = np.random.choice(awareness_templates)
        risk_text = np.random.choice(risk_templates)
        
        # Combine into full text
        full_text = f"{green_text} {awareness_text} {risk_text}"
        texts.append(full_text)
        
        # Extract scores (for validation)
        green_scores.append(green_perception_templates.index(green_text) + 1)
        awareness_scores.append(awareness_templates.index(awareness_text) + 1)
        risk_scores.append(risk_templates.index(risk_text) + 1)
    
    df = pd.DataFrame({
        'text': texts,
        'green_perception': green_scores,
        'environmental_awareness': awareness_scores,
        'perceived_risk': risk_scores
    })
    
    return df

