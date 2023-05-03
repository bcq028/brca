import pandas as pd
from scipy.stats import ttest_ind
import os

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

    if method=='t_FDR':
        # Calculate t-FDR and get significant genes
        t_stat, p_val = ttest_ind(normal_matrix, tumor_matrix)
        fdr = p_val.shape[0] * p_val / p_val.argsort()[::-1]
        significant_genes = normal_data.index[fdr < 0.05]

        # Subset data to significant genes
        normal_matrix_sub = normal_data.loc[significant_genes]
        tumor_matrix_sub = tumor_data.loc[significant_genes]
    elif method=='random':
        normal_matrix_sub = data['normal']
        tumor_matrix_sub = data['tumor']
        #TODO

    if not os.path.exists('reduction'):
        os.makedirs('reduction')

    pd.DataFrame(normal_matrix_sub).to_csv('reduction/normal_matrix_sub.csv', index=True)
    pd.DataFrame(tumor_matrix_sub).to_csv('reduction/tumor_matrix_sub.csv', index=True)

def sort_by_importance():
    #TODO
    pass 

def main():
    dim_reduction("t_FDR")
    sort_by_importance()

main()