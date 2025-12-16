"""
Prepare UCI Energy Efficiency Dataset for our pipeline
Maps UCI dataset columns to our pipeline's expected structure
"""

import pandas as pd
import numpy as np

def prepare_uci_dataset(input_file='data/ENB2012_data.xlsx', output_file='data/uci_energy_efficiency.csv'):
    """
    Transform UCI Energy Efficiency Dataset to match our pipeline structure
    
    UCI Dataset Columns:
    - X1: Relative Compactness
    - X2: Surface Area
    - X3: Wall Area
    - X4: Roof Area
    - X5: Overall Height
    - X6: Orientation (2-5)
    - X7: Glazing Area (0, 0.1, 0.2, 0.3, 0.4, 0.5)
    - X8: Glazing Area Distribution (0-5)
    - Y1: Heating Load (target)
    - Y2: Cooling Load (target)
    """
    
    print("Loading UCI Energy Efficiency Dataset...")
    df = pd.read_excel(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Create new DataFrame with mapped columns
    df_new = pd.DataFrame()
    
    # Map to our pipeline's expected features
    # Building Design Parameters
    df_new['orientation'] = df['X6'].astype(int)  # Orientation (2-5)
    df_new['roof_type'] = (df['X8'] > 0).astype(int)  # Derived from glazing distribution
    df_new['insulation'] = df['X1']  # Relative compactness (related to insulation)
    
    # Create material type based on compactness
    df_new['material'] = pd.cut(df['X1'], bins=[0, 0.5, 0.7, 1.0], 
                                labels=['wood', 'composite', 'concrete'])
    df_new['material'] = df_new['material'].astype(str)
    
    # Building dimensions (for feature engineering)
    df_new['surface_area'] = df['X2']
    df_new['wall_area'] = df['X3']
    df_new['roof_area'] = df['X4']
    df_new['height'] = df['X5']
    df_new['glazing_area'] = df['X7']
    df_new['glazing_distribution'] = df['X8']
    
    # IoT Sensor Data (synthetic but realistic based on building characteristics)
    np.random.seed(42)
    # Temperature based on heating/cooling load
    df_new['temperature'] = 20 + (df['Y1'] + df['Y2']) / 10 + np.random.normal(0, 2, len(df))
    # Humidity inversely related to efficiency
    df_new['humidity'] = 50 - df['X1'] * 20 + np.random.normal(0, 5, len(df))
    df_new['humidity'] = np.clip(df_new['humidity'], 20, 80)
    # Air quality based on glazing
    df_new['air_quality'] = 50 + df['X7'] * 50 + np.random.normal(0, 10, len(df))
    df_new['air_quality'] = np.clip(df_new['air_quality'], 0, 100)
    # Equipment status (0=off, 1=on, 2=maintenance)
    df_new['equipment_status'] = np.random.choice([0, 1, 2], len(df), p=[0.1, 0.8, 0.1])
    
    # Climate Data (synthetic but realistic)
    # Solar radiation based on orientation and glazing
    orientation_factor = df['X6'].map({2: 0.8, 3: 1.0, 4: 0.9, 5: 0.7})
    df_new['solar_radiation'] = 300 + orientation_factor * 200 * (1 + df['X7']) + np.random.normal(0, 50, len(df))
    df_new['solar_radiation'] = np.clip(df_new['solar_radiation'], 100, 800)
    # Wind speed (random but realistic)
    df_new['wind_speed'] = np.random.uniform(2, 12, len(df))
    # Temperature profile (average temperature)
    df_new['temperature_profile'] = df_new['temperature'] + np.random.normal(0, 3, len(df))
    
    # Consumer Survey Data (synthetic but correlated with efficiency)
    # Green perception based on efficiency
    efficiency_score = 1 - (df['Y1'] + df['Y2']) / 100
    df_new['green_perception'] = np.clip(efficiency_score * 5, 1, 5) + np.random.normal(0, 0.5, len(df))
    df_new['green_perception'] = np.clip(df_new['green_perception'], 1, 5)
    # Environmental awareness
    df_new['environmental_awareness'] = df_new['green_perception'] + np.random.normal(0, 0.3, len(df))
    df_new['environmental_awareness'] = np.clip(df_new['environmental_awareness'], 1, 5)
    # Perceived risk (inverse of efficiency)
    df_new['perceived_risk'] = 6 - df_new['green_perception'] + np.random.normal(0, 0.3, len(df))
    df_new['perceived_risk'] = np.clip(df_new['perceived_risk'], 1, 5)
    
    # Stakeholder and Operational Data
    df_new['metro_logistics'] = np.random.uniform(0.6, 1.0, len(df))
    df_new['policy_compliance'] = 0.7 + df_new['green_perception'] / 10 + np.random.normal(0, 0.1, len(df))
    df_new['policy_compliance'] = np.clip(df_new['policy_compliance'], 0.5, 1.0)
    
    # Target Variables
    # Create energy efficiency score (0-1) from heating and cooling loads
    # Lower loads = higher efficiency
    max_load = df[['Y1', 'Y2']].max().max()
    min_load = df[['Y1', 'Y2']].min().min()
    total_load = df['Y1'] + df['Y2']
    df_new['energy_efficiency'] = 1 - (total_load - min_load * 2) / (max_load * 2 - min_load * 2)
    df_new['energy_efficiency'] = np.clip(df_new['energy_efficiency'], 0, 1)
    
    # Also keep original targets for reference
    df_new['heating_load'] = df['Y1']
    df_new['cooling_load'] = df['Y2']
    df_new['energy_consumption'] = total_load * 10  # Convert to kWh scale
    
    # Add timestamp for temporal analysis (optional)
    import datetime
    start_date = datetime.datetime(2020, 1, 1)
    df_new['timestamp'] = [start_date + datetime.timedelta(days=i, hours=np.random.randint(0, 24)) 
                           for i in range(len(df_new))]
    
    # Reorder columns to match expected structure
    feature_cols = [
        'temperature', 'humidity', 'air_quality', 'equipment_status',  # IoT
        'orientation', 'material', 'roof_type', 'insulation',  # Design
        'green_perception', 'environmental_awareness', 'perceived_risk',  # Survey
        'solar_radiation', 'wind_speed', 'temperature_profile',  # Climate
        'metro_logistics', 'policy_compliance',  # Operational
        'timestamp',  # Temporal
        'energy_efficiency', 'energy_consumption', 'heating_load', 'cooling_load'  # Targets
    ]
    
    # Select only columns that exist
    available_cols = [col for col in feature_cols if col in df_new.columns]
    df_final = df_new[available_cols]
    
    # Save to CSV
    df_final.to_csv(output_file, index=False)
    
    print(f"\nPrepared dataset saved to: {output_file}")
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")
    print(f"\nEnergy Efficiency Statistics:")
    print(df_final['energy_efficiency'].describe())
    print(f"\nFirst few rows:")
    print(df_final.head())
    
    return df_final

if __name__ == '__main__':
    prepare_uci_dataset()

