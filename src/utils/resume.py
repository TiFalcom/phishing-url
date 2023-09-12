import pandas as pd
import numpy as np

def resume_table(df, target=None):
    
    if target != None:
        print(f'Badrate: {df[target].value_counts(normalize=True)[1]}')
        
    print(f"Formato dataset: {df.shape}")
    summario = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summario = summario.reset_index()
    summario['Nome'] = summario['index']
    summario = summario[['Nome', 'dtypes']]
    summario['Missing'] = df.isnull().sum().values
    summario['Unicos'] = df.nunique().values
    summario['Primeiro Valor'] = df.loc[0].values
    summario['Segundo Valor'] = df.loc[1].values
    return summario