import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from feature_engine.encoding import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

class Correlation():
    def __init__(self, vars_cat=[], vars_num=[], fig_size=(10,10)):
        self.vars_cat = vars_cat
        self.vars_num = vars_num
        self.fig_size = fig_size
    
    def __encoding_cats(self, X):
        encOrd = OrdinalEncoder(encoding_method='arbitrary')
        X_tmp = encOrd.fit_transform(X[self.vars_cat])
        return X_tmp
    
    def categorical_correlation(self, X, y, max_cor=0.9):
        
        X_tmp = self.__encoding_cats(X)
        self.vars_cat = self.best_features(X_tmp, y, self.vars_cat)
        
        corr_matrix = X_tmp[self.vars_cat].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        vars_remove = [column for column in upper.columns if any(upper[column] > max_cor)]
        
        return vars_remove
    
    def numerical_correlation(self, X, y, max_cor):
        
        self.vars_num = self.best_features(X, y, self.vars_num)
        corr_matrix = X[self.vars_num].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        vars_remove = [column for column in upper.columns if any(upper[column] > max_cor)]
        
        return vars_remove
    
    def best_features(self, X, y, variables):
        #ANOVA
        slkb = SelectKBest(score_func=f_classif, k='all').fit(X[variables], y)
        _ , best_vars = zip(*sorted(zip(slkb.scores_, slkb.get_feature_names_out()), reverse=True))
        
        return list(best_vars)
    
    def remove_correlation(self, X, y, max_cor=0.9, mode='both'):
        
        if mode == 'both':
            vars_remove = self.numerical_correlation(X, y, max_cor)
            vars_remove = vars_remove + self.categorical_correlation(X, y, max_cor)
            
        elif mode == 'num':
            vars_remove = self.numerical_correlation(X, y, max_cor)
            
        elif mode == 'cat':
            vars_remove = self.categorical_correlation(X, y, max_cor)
            
        return vars_remove
    
    def plot(self, X, mode='both'):
        
        if mode == 'both':     
            X_tmp = self.__encoding_cats(X)
            corr_cat = X_tmp[self.vars_cat].corr(method='spearman')
            corr_num = X[self.vars_num].corr(method='pearson')
            
            fig, axs = plt.subplots(2, 1, figsize=self.fig_size)
            
            sns.heatmap(corr_cat, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[0])
            axs[0].set_title('Heatmap de Correlação Spearman Categóricas')
            
            sns.heatmap(corr_num, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=axs[1])
            axs[1].set_title('Heatmap de Correlação Pearson Numéricas')
            
        elif mode == 'cat':
            X_tmp = self.__encoding_cats(X)
            corr_cat = X_tmp[self.vars_cat].corr(method='spearman')
            
            plt.figure(figsize=self.fig_size)
            sns.heatmap(corr_cat, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Heatmap de Correlação Spearman Categóricas')
        
        elif mode == 'num':
            corr_num = X[self.vars_num].corr(method='pearson')
        
            plt.figure(figsize=self.fig_size)
            sns.heatmap(corr_num, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Heatmap de Correlação Pearson Numéricas')
        
        plt.show()