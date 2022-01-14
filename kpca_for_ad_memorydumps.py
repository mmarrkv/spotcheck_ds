'''
Feature scaling: attribute ratio where each row adds up to 1; Rending each row execution sample length insensitive
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle

'''
print("place quick tests here")
sys.exit(0)
'''

#Datase load + scale

def normalize(x, rowsum, histogram_dim):

    if rowsum > 0:
        return x/rowsum
    else:
        return 1/histogram_dim #Handles all-0 vector case

def rowratio(row): #histogram scaling to row ratio is insensitive "overall" to execution time + HANDLES probs with all-0 vectors
    histogram_dim = row.shape[0]    
    return row.apply(normalize, args=(row.sum(),histogram_dim))

#Datase load + scale
def get_dataset(file_path):

    df = pd.read_csv(file_path)

    selectcols = df.loc[:, 'apk_AccessibilityManager':'apk_WindowManager']
    selectcols = selectcols.astype(np.float64)
    histogram_dim = selectcols.shape[1]
    selectcols=selectcols.apply(rowratio, axis=1)
    labels = df.loc[:, 'apk_id':'apk_is_mal']
    return selectcols, labels

#Grid Search cross-validation scorer
def my_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)


#latent visualization
def vis_latent(data, labels,
               model_name="kpca_for_ad_hprof"):

    y_true = labels['apk_is_mal']
    filename = os.path.join(model_name, "kpca_mean_hprof.png")

    plt.figure(figsize=(12, 10))
    plt.scatter(data[:, 0], data[:, 1], c=y_true)
    plt.colorbar(ticks=[0, 1])
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    help_ = "Load trained principal components"
    optional.add_argument("-w", "--weights", help=help_)
    help_ = "Path to train+val/test csv"
    required.add_argument('-f', '--fpath', help=help_, required=True)
    args = parser.parse_args()


    if args.weights:

        print("Using trained model..")
        rbf_pca = pickle.load(open(args.weights, 'rb'))

        df, labels = get_dataset(args.fpath)
        print(df.head())

        X = df.to_numpy()


        X_reduced = rbf_pca.transform(X)
        vis_latent(X_reduced,labels)
        X_preimage = rbf_pca.inverse_transform(X_reduced)


        norows = X.shape[0]
        reconerr_scores = np.zeros([norows])
        for i in range(0,norows):
            mse=mean_squared_error(X[i], X_preimage[i])
            reconerr_scores[i]=mse
            #print("mse "+str(i)+" "+str(mse))
 
        print(reconerr_scores)
        labels['ReconErr_Scores'] =  reconerr_scores
        labels.to_csv ('kpca_for_ad_hprof/kpca_for_ad_hprof_scores.csv', index = False, header=True)


    else:

        print('Training mode...')
        os.makedirs('kpca_for_ad_hprof', exist_ok=True)

        df, labels = get_dataset(args.fpath)
        print(df.head())
        X = df.to_numpy()
        print(X[:2][:])

        param_grid = [{"gamma": np.linspace(0.03, 0.05, 10), "kernel": ["rbf"] }]
        rbf_pca = KernelPCA(n_components = 2, kernel="rbf", fit_inverse_transform=True)
        grid_search = GridSearchCV(rbf_pca, param_grid, cv=3, scoring=my_scorer)
        grid_search.fit(X)

        X_reduced = grid_search.transform(X)
        print(X_reduced[:2][:])

        X_preimage = grid_search.inverse_transform(X_reduced)
        print(X_preimage[:2][:])

        filename = "kpca_for_ad_hprof/kpca_for_ad_hprof.mdl"
        pickle.dump(grid_search, open(filename, 'wb'))





