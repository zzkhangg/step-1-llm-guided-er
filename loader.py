import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')

    except Exception as e:
        return f"Error reading file with pandas: {e}"
    
    return df