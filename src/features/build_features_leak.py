import sys
sys.path.append('./src')
from utils.transformers import BuildFeatures, BuildFeaturesEmbedding, BuildFeaturesEmbeddingLeak
import logging
import os
import gc
import pickle
import pandas as pd
import scipy.sparse as sp


def main():
    
    logging.info('Lendo tabela')
    
    df = pd.read_csv(os.path.join('data','phishing_site_urls.csv'))
    
    logging.info(f'Tabela lida. Shape: {df.shape}')
    
    logging.info('Iniciando processamento de embedding')
    
    bfe = BuildFeaturesEmbedding()
    
    bfel = BuildFeaturesEmbeddingLeak()
    
    df1 = bfe.fit_transform(df)
    
    df1 = bfel.fit_transform(df1)
    
    logging.info(f'Processamento basico finalizado. Shape: {df1.shape}')
    
    logging.info('Salvando tabela basica')
    
    sp.save_npz('data/embedding_processed/embedding.npz', df1)
    
    logging.info('Tabela basica salva com sucesso')
    
    logging.info('Salvando binarios')
    
    pickle.dump(bfe, open('model/encoders/embedding_features.pkl', 'wb'))
    
    pickle.dump(bfel, open('model/encoders/embedding_features_leak.pkl', 'wb'))
    
    logging.info('Binarios salvos com sucesso')
    
    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
    
    main()