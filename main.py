import pandas as pd
from scipy.stats import ttest_ind
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from os import listdir
SUPPORTED_METHODS=["t_FDR","random_forest"]

def dim_reduction(method:str):
    # Load data
    normal_data = pd.read_csv('row/normal_matrix.csv', index_col=0)
    tumor_data = pd.read_csv('row/tumor_matrix.csv', index_col=0)

    # Convert data to numpy arrays and transpose
    normal_matrix = normal_data.to_numpy().T
    tumor_matrix = tumor_data.to_numpy().T

    if not method in SUPPORTED_METHODS:
        raise ValueError("Unsupported method")

    normal_matrix_sub=normal_data
    tumor_matrix_sub=tumor_data
    if method=='t_FDR':
        # Calculate t-FDR and get significant genes
        t_stat, p_val = ttest_ind(normal_matrix, tumor_matrix)
        fdr = p_val.shape[0] * p_val / p_val.argsort()[::-1]
        significant_genes = normal_data.index[fdr < 0.05]

        # Subset data to significant genes
        normal_matrix_sub = normal_data.loc[significant_genes]
        tumor_matrix_sub = tumor_data.loc[significant_genes]
    elif method=='random_forest':
        rf = RandomForestClassifier(n_estimators=10,max_depth=10)
        rf.fit(np.concatenate((normal_matrix,tumor_matrix)), [0] * normal_matrix.shape[0] + [1] * tumor_matrix.shape[0])
        importances = rf.feature_importances_
    
        # Calculate mean decrease impurity (MDI)
        mdi = np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    
        # Calculate the decrease impurity for each feature
        mdi_values = np.empty(importances.shape[0])
        for feature in range(importances.shape[0]):
            if feature%5000==0: print(f'\rin process...{"{:.2f}".format(feature/importances.shape[0]*100)}%')
            mdi_values[feature] = np.mean([tree.feature_importances_[feature] - mdi[feature] for tree in rf.estimators_])
        # Calculate the mean of MDI values
        mdi_mean = np.mean(mdi_values)
    
        # Subset data to features with MDI values above the mean
        significant_genes1 = normal_data.index[mdi_values >= mdi_mean]
        significant_genes2= tumor_data.index[mdi_values>= mdi_mean]
    
        # Subset data to significant genes
        normal_matrix_sub = normal_data.loc[significant_genes1,:]
        tumor_matrix_sub = tumor_data.loc[significant_genes2,:]

    if not os.path.exists('reduction'):
        os.makedirs('reduction')

    pd.DataFrame(normal_matrix_sub).to_csv(f'reduction/normal_matrix_{method}.csv', index=True)
    pd.DataFrame(tumor_matrix_sub).to_csv(f'reduction/tumor_matrix_{method}.csv', index=True)

def sort_by_importance():
    #TODO
    pass 

def main():
   if 'normal_matrix_t_FDR.csv' not in listdir('reduction'):
    print('generating FDR reduction...')
    dim_reduction("t_FDR")
   if 'normal_matrix_random_forest.csv' not in listdir('reduction'):
    print('generating random_forest reduction...')
    dim_reduction("random_forest")
    sort_by_importance()

main()