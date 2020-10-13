import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import subprocess as sp
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import circmean
from lds_regression import smooth_spikes
from predict_targets import *

if __name__=='__main__':

    n_splits = 2
    performance = np.zeros(n_splits)

    processed_inputs = snakemake.input[0]

    df = pd.read_pickle(processed_inputs)

    X_input = np.vstack(df['input'].values)
    dt = 0.001
    spike_dt = 0.01
    spikes = [np.array(s) for s in df['spikes'].values]
    smoothed,_ = smooth_spikes(spikes, dt, spike_dt, std=.020)
    n_targets = len(smoothed)
    n_win = smoothed[0].shape[0]
    n_neuron_components = 2
    neuron_pca = PCA(n_components=n_neuron_components)
    smoothed = neuron_pca.fit_transform(np.concatenate(smoothed))
    X_input = smoothed.reshape((n_targets, n_win*n_neuron_components))
    
    pca_input = PCA(n_components=20)
    pca_input.fit(X_input)
    X_input = pca_input.transform(X_input)
    X_input = np.hstack((df[['peak_input_1','peak_input_2']].values,X_input))

    pca_kin = PCA(n_components=5)
    kin = np.vstack(df['kinematics'].values)
    pca_kin.fit(kin)
    kin = pca_kin.transform(kin)

    metrics = ['r^2', 'Mean distance (mm)', 'Categorical Accuracy (%)']
    include_kins = ['LFADS predictor', 'Kinematic + LFADS predictor']
    references = ['Allocentric', 'Hand-centric']

    classifiers = [',\n '.join([kin, ref]) for kin in include_kins for ref in references]

    commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
    with PdfPages(snakemake.output[0], metadata={'commit':commit}) as pdf:
        for fig_idx, metric in enumerate(metrics):
            av_performance = []
            for include_kin in include_kins:
                if include_kin == 'Kinematic + LFADS predictor':
                    X = np.hstack((X_input, kin))
                else:
                    X = np.copy(X_input)

                for reference in references:
                    # if include_kin == 'Kinematic + LFADS predictor' and reference == 'Hand-centric':
                    #     continue

                    if reference == 'Hand-centric':
                        Y = df[['target_x', 'target_y']].values - df[['x','y']].values
                    elif reference == 'Allocentric':
                        Y = df[['target_x', 'target_y']].values
                    # Y[:,0] -= np.min(Y[:,0])
                    # # Y[:,1] -= np.min(Y[:,1])
                    # Y = np.sqrt(Y)
                    kf=KFold(n_splits=n_splits, shuffle=True, random_state=0)
                    
                    i = 0
                    for train_idx, test_idx in kf.split(X):
                        X_train, X_test = X[train_idx], X[test_idx]
                        Y_train, Y_test = Y[train_idx], Y[test_idx]
                    
                        if metric == 'r^2':       
                            model = RandomForestRegressor()
                            model.fit(X_train, Y_train)
                            prediction = model.predict(X_test)             
                            #performance[i] = np.corrcoef(prediction.flatten(), Y_test.flatten())[1,0]**2
                            performance[i] = 1 - np.sum((prediction.flatten() - Y_test.flatten())**2)/prediction.size/np.var(Y_test.flatten())
                        elif metric == 'Mean distance (mm)':
                            model = RandomForestRegressor()
                            model.fit(X_train, Y_train)
                            prediction = model.predict(X_test)
                            performance[i] = np.mean(np.linalg.norm(Y_test-prediction,axis=1))
                        elif metric == 'Categorical Accuracy (%)':
                            cat_model = RandomForestClassifier()
                            cat_model.fit(X_train, bin_y(Y_train))
                            cat_prediction = cat_model.predict(X_test)
                            accuracy = np.sum(cat_prediction == bin_y(Y_test))/Y_test.shape[0]
                            performance[i] = accuracy
                        i += 1

                    av_performance.append(np.mean(performance))
            fig = plt.figure(figsize=(12,4))
            plt.plot(av_performance, 'r.-')
            plt.title('Target Prediction')
            plt.xlabel('Classifier')
            plt.ylabel(metric)
            plt.xticks(ticks=np.arange(len(av_performance)), labels=classifiers, fontsize=10)
            pdf.savefig(fig)