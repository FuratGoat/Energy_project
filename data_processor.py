import pandas as pd
import json

def load_and_process_data():
    """Load and process the energy data"""
    try:
        with open('energy_data.json', 'r') as f:
            data = json.load(f)
        
        if 'response' not in data or 'data' not in data['response']:
            raise ValueError("Invalid data format")
        
        # Create initial dataframe
        df = pd.DataFrame(data['response']['data'])
        
        # Filter for industrial sector
        df_industrial = df[df["sectorName"] == "industrial"].copy()
        
        # Convert data types
        df_industrial['price'] = pd.to_numeric(df_industrial['price'], errors='coerce')
        df_industrial['period'] = pd.to_datetime(df_industrial['period'], errors='coerce')
        
        # Create pivot table
        df_state = df_industrial.pivot_table(
            values='price', 
            index='period', 
            columns='stateDescription',
            aggfunc='first'
        ).sort_index()  # Sort by date
        
        # Forward fill missing values within a reasonable limit
        df_state = df_state.fillna(method='ffill', limit=2)
        
        return df_state
    
    except Exception as e:
        print(f"Error in load_and_process_data: {str(e)}")
        return None
