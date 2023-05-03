import pandas as pd
from scipy.stats import ttest_ind
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from os import listdir
from sklearn import svm
from sklearn.metrics import confusion_matrix

SUPPORTED_REDUCE_METHODS=["t_FDR","random_forest"]
SUPPORTED_SORT_METHODS=["random_forest","SVM"]

def dim_reduction(method:str):
    # Load data
    normal_data = pd.read_csv('row/normal_matrix.csv', index_col=0)
    tumor_data = pd.read_csv('row/tumor_matrix.csv', index_col=0)

    # Convert data to numpy arrays and transpose
    normal_matrix = normal_data.to_numpy().T
    tumor_matrix = tumor_data.to_numpy().T

    if not method in SUPPORTED_REDUCE_METHODS:
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


# 计算变量重要性
def calculate_importance_rf(normal_data, tumor_data):
    # 合并两个数据集
    data = pd.concat([normal_data, tumor_data],axis=1).T

    # 构建标签
    labels = np.array([0]*normal_data.shape[1] + [1]*tumor_data.shape[1])

    # 训练随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(data, labels)

    # 获取变量重要性
    importances = rf.feature_importances_
    return normal_data.index.values, importances

def calculate_importance_svm(normal_data, tumor_data):
    # 将正常数据和肿瘤数据合并成一个数据集
    data = np.concatenate((normal_data, tumor_data),axis=1).T
    
    # 创建标签数组（正常标签为0，肿瘤标签为1）
    labels = np.array([0]*normal_data.shape[1] + [1]*tumor_data.shape[1])
    
    # 使用线性SVM训练分类器
    clf = svm.SVC(kernel='linear')
    clf.fit(data, labels)
    
    importances = np.abs(clf.coef_[0])
    
    return normal_data.index.values, importances

# Holdout 验证
def holdout_validation(normal_data, tumor_data):
    # 将数据分割成训练集和测试集，2/3 为训练集，1/3 为测试集
    normal_train = normal_data.sample(frac=2/3, random_state=42)
    normal_test = normal_data.drop(normal_train.index)
    tumor_train = tumor_data.sample(frac=2/3, random_state=42)
    tumor_test = tumor_data.drop(tumor_train.index)

    # 构建标签
    y_train = np.array([0]*len(normal_train) + [1]*len(tumor_train))
    y_test = np.array([0]*len(normal_test) + [1]*len(tumor_test))

    # 训练随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(pd.concat([normal_train, tumor_train]), y_train)

    # 在测试集上进行预测
    y_pred = rf.predict(pd.concat([normal_test, tumor_test]))

    # 计算指标
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tn+tp) / (tn+fp+fn+tp)
    sen = tp / (tp+fn)
    spec = tn / (tn+fp)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    f1 = 2*tp / (2*tp+fp+fn)
    jaccard = tp / (tp+fp+fn)

    return acc, sen, spec, ppv, npv, f1, jaccard

def sort_by_importance(reductType:str, sortType:str):
    if sortType not in SUPPORTED_SORT_METHODS or reductType not in SUPPORTED_REDUCE_METHODS:
        print("not supported")

    normal_data = pd.read_csv(f'reduction/normal_matrix_{reductType}.csv', index_col=0)
    tumor_data = pd.read_csv(f'reduction/tumor_matrix_{reductType}.csv', index_col=0)
    
    features, importances = (calculate_importance_rf(normal_data, tumor_data) if sortType=='random_forest'
    else calculate_importance_svm(normal_data, tumor_data) if sortType=='SVM' else (np.empty((0, 3)),np.empty((0, 3)))
    )

    if np.array_equal(features, np.empty((0,3))):
        raise ValueError("unsupport type")

    feature_importance_sorted = np.argsort(importances)[::-1]
    results = []
    for i in range(1, 11):
        selected_features = features[feature_importance_sorted[:i]]
        acc, sen, spec, ppv, npv, f1, jaccard = holdout_validation(normal_data.T[selected_features],
                                                                   tumor_data.T[selected_features])
        results.append([i, importances[feature_importance_sorted[i-1]], acc, sen, spec, ppv, npv, f1, jaccard])

    print(f'{reductType}-{sortType} method 10 most important gene: {normal_data.index[feature_importance_sorted[:10]].tolist()}')


    # 将结果转换成DataFrame格式并保存
    columns = ['num_features', 'importance_score', 'accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1', 'jaccard']
    results_df = pd.DataFrame(results, columns=columns)
    if not os.path.exists('results'):
        os.makedirs('results')
    results_df.to_csv(f'results/{reductType}-{sortType}.csv')

    

def main():
    if 'normal_matrix_t_FDR.csv' not in listdir('reduction'):
        print('generating FDR reduction...')
        dim_reduction("t_FDR")
    if 'normal_matrix_random_forest.csv' not in listdir('reduction'):
        print('generating random_forest reduction...')
        dim_reduction("random_forest")

    sort_by_importance('t_FDR','random_forest')
    sort_by_importance('t_FDR','SVM')
    sort_by_importance('random_forest','random_forest')
    sort_by_importance('random_forest','SVM')

main()