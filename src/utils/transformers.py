import os

import pandas as pd
import numpy as np
import re
import scipy.sparse as sp

import nltk
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


def get_filename_extension(text):
    file = os.path.basename(text)
    
    if '.' not in file:
        return 'NULL'
    
    return file.split('.')


def is_valid_ip(domain):
    ip_pattern = r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    
    if re.match(ip_pattern, domain):
        return 1
    else:
        return 0
    
    
def get_qty_features(X, cols):
    
    for col in cols:
        #Len
        X[f'len_{col}'] = X[col].str.len()
        #Dots
        X[f'qty_dots_{col}'] = X[col].str.count('.')
        #Hyphen
        X[f'qty_hyphens_{col}'] = X[col].str.count('-')
        #Underscore
        X[f'qty_undescore_{col}'] = X[col].str.count('_')
        #Numbers
        X[f'qty_numbers_{col}'] = X[col].str.count('\b')
        #Vogais
        X[f'qty_vogais_{col}'] = X[col].str.count(r'[aeiouAEIOU]')
        #Especiais
        X[f'qty_especiais_{col}'] = X[col].str.count(r'[!@#$%^&*()_+]')
        
        
def get_file_name_extension(X):
    X['file'] = X['URL'].apply(os.path.basename)
    
    X['file'] = np.where(X['file'].str.contains('\.'),X['file'], '')
    
    X[['file_name', 'file_extension']] = X['file'].str.split('.', n=1, expand=True).fillna('')
    
    
def get_valid_words(text, tokenizer):
    text = tokenizer.tokenize(text)
    
    result = ' '.join(text)
    
    return result


class BuildFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X_tmp = X.reset_index(drop=True)
        
        #Domain
        X_tmp['domain'] = X_tmp['URL'].str.split("/", n=1).str[0].fillna('')
        X_tmp['domain_splited'] = X_tmp['domain'].str.split('\.')
        
        X_tmp['org_domain'] = X_tmp['domain_splited'].apply(lambda x: 1 if 'org' in x else 0)
        X_tmp['com_domain'] = X_tmp['domain_splited'].apply(lambda x: 1 if 'com' in x else 0)
        X_tmp['gov_domain'] = X_tmp['domain_splited'].apply(lambda x: 1 if 'gov' in x else 0)
        
        #Query
        X_tmp['query'] = X_tmp['URL'].str.split('\?').str[1].fillna('')
        X_tmp['qtd_args_query'] = X_tmp['query'].str.count('=')
        
        #File
        get_file_name_extension(X_tmp)
        
        #General Features
        get_qty_features(X_tmp, ['URL', 'domain', 'query', 'file', 'file_name', 'file_extension'])
        
        X_tmp['have_domain'] = np.where(X_tmp['domain'] == '', 0, 1)
        X_tmp['have_query'] = np.where(X_tmp['query'] == '', 0, 1)
        X_tmp['have_file'] = np.where(X_tmp['file'] == '', 0, 1)
        
        return X_tmp

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    
class BuildFeaturesEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y=None):
        
        self.tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        
        return self
    
    def transform(self, X):
        
        X_tmp = X.reset_index(drop=True)
        
        X_tmp['domain'] = X_tmp['URL'].str.split("/", n=1).str[0].fillna('')
        
        X_tmp['tokenized_domain'] = X_tmp['domain'].map(lambda text: get_valid_words(text, self.tokenizer))
        
        X_tmp['tokenized_total'] = X_tmp['URL'].map(lambda text: get_valid_words(text, self.tokenizer))
        
        return X_tmp
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    
class BuildFeaturesEmbeddingLeak(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):        
        self.cv_domain = CountVectorizer(decode_error='ignore').fit(X['tokenized_domain'])
        self.cv_total = CountVectorizer(decode_error='ignore').fit(X['tokenized_total'])
        
    def transform(self, X):
        
        matrix_domain = self.cv_domain.transform(X['tokenized_domain'])
        
        matrix_total = self.cv_total.transform(X['tokenized_total'])
        
        X_tmp = sp.hstack([matrix_domain, matrix_total])
        
        return X_tmp
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)