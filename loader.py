import pandas as pd

def load_data(file_path, encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python',encoding=encoding)

    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return df