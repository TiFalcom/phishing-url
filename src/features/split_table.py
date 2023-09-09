import sys
sys.path.append('./src')
from utils.transformers import BuildFeatures, BuildFeaturesEmbedding, BuildFeaturesEmbeddingLeak
from sklearn.model_selection import train_test_split
import logging
import os
import gc
import pickle
import pandas as pd
import scipy.sparse as sp


def main():
    
    logging.info('Lendo tabela basica')
    
    df = pd.read_parquet(os.path.join('data', 'basic_processed', 'basic.parquet.gzip'))
    
    logging.info(f'Tabela lida. Shape: {df.shape}')
    
    logging.info('Iniciando separacao de treino e teste')
    
    treino, teste = train_test_split(df, test_size=0.7, random_state=777, stratify=df['Label'])
    
    logging.info(f'Treino Shape: {treino.shape}. Teste Shape: {teste.shape}')
    
    logging.info('Salvando tabela basica')
    
    treino.to_parquet('data/train_test/train_basic.parquet.gzip', compression='gzip', index=False)
    
    teste.to_parquet('data/train_test/test_basic.parquet.gzip', compression='gzip', index=False)
    
    logging.info('Tabela basica salva com sucesso')
    
    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    main()