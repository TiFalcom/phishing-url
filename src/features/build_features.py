import sys
sys.path.append('./src')
from utils.transformers import BuildFeatures, BuildFeaturesEmbedding, BuildFeaturesEmbeddingLeak
import logging
import os
import gc
import pickle
import pandas as pd


def main():
    
    logging.info('Lendo tabela')
    
    df = pd.read_csv(os.path.join('data','phishing_site_urls.csv'))
    
    logging.info(f'Tabela lida. Shape: {df.shape}')
    
    logging.info('Iniciando processamento basico')
    
    bf = BuildFeatures()
    
    df1 = bf.fit_transform(df)
    
    logging.info(f'Processamento basico finalizado. Shape: {df1.shape}')
    
    logging.info('Salvando tabela basica')
    
    df1.to_parquet('data/basic_processed/basic.parquet.gzip', compression='gzip', index=False)
    
    logging.info('Tabela basica salva com sucesso')
    
    logging.info('Salvando binarios')
    
    pickle.dump(bf, open('model/encoders/basic_features.pkl', 'wb'))
    
    logging.info('Binarios salvos com sucesso')
    
    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    main()