import pandas as pd
import numpy as np
import math
import random
import argparse
import os

from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from datetime import datetime
import pickle

#import torch
#from torch import nn 
import matplotlib.pyplot as plt

# custom functions
from bin import nested_cv_rfe, compare_hyperparameters, evaluate, rfe_model, svm_model, get_train_dev_test_splits, dataframe_to_vecs, aggregate_data, clean_data


parser = argparse.ArgumentParser(description="Run Russian dyslexia baseline")
parser.add_argument("--subjpred", dest="batch_subjects", action="store_true")
parser.add_argument("--textpred", dest="batch_subjects", action="store_false")
parser.add_argument("--grade", dest="grade", action="store_true")
parser.add_argument("--nograde", dest="grade", action="store_false")
parser.set_defaults(batch_subjects=True)
parser.set_defaults(grade=True)
args = parser.parse_args()


np.random.seed(42)
random.seed(42)
BATCH_SUBJECTS = args.batch_subjects
GRADE = args.grade

fig, ax = plt.subplots()

# read in the data file
if GRADE:
    file = 'data/baseline_dataset_grade.csv'
    grade_setting = 'grade'
else:
    file = 'data/baseline_dataset.csv'
    grade_setting = 'nograde'
    
if BATCH_SUBJECTS:
    setting = "reader"
    folder = "reader_prediction_results/"
else:
    setting = "sentence"
    folder = "sentence_prediction_results/"

dirExist = os.path.exists(folder)
if not dirExist:
    os.makedirs(folder)

    
sacc_file = pd.read_csv(file) 
data_df = clean_data(sacc_file, BATCH_SUBJECTS)

dyslexic_subjects = data_df[data_df["group"] == 1]["subj"].unique()
control_subjects = data_df[data_df["group"] == 0]["subj"].unique()

# aggregate data: calculate mean and std for each ET measure
feature_group_df = aggregate_data(data_df, BATCH_SUBJECTS)
feature_group_df = feature_group_df.dropna()

NUM_FOLDS = 10
# shuffle and distribute on stratified folds
n_folds = NUM_FOLDS
folds = [[] for _ in range(n_folds)]
random.shuffle(dyslexic_subjects)
random.shuffle(control_subjects)
for i, subj in enumerate(dyslexic_subjects):
    folds[i % n_folds].append(subj)
for i, subj in enumerate(control_subjects):
    folds[n_folds - 1 - i % n_folds].append(subj)
for fold in folds:
    random.shuffle(fold)
    
    
# vectorize labels and features
feature_folds, label_folds = dataframe_to_vecs(feature_group_df, folds, BATCH_SUBJECTS)
    
n_features = 26
c_params = [0.1, 0.5, 1, 5, 10, 50, 100, 500]  


# run cross validation with recursive feature selection
final_scores, final_scores_std, best_params, best_n_features, CV_eval, saving_scores, aucs = nested_cv_rfe(feature_folds, 
                                                                    label_folds, n_folds, c_params,
                                                                    n_features, ax)

    
with open(f"{folder}aucs_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(aucs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f"{folder}CV_eval_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(CV_eval, handle, protocol=pickle.HIGHEST_PROTOCOL) 

with open(f"{folder}label_folds_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(label_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f"{folder}best_n_features_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(best_n_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f"{folder}saving_scores_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(saving_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f"{folder}best_params_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f"{folder}folds_{setting}_{grade_setting}.pickle" , 'wb') as handle:
    pickle.dump(folds, handle, protocol=pickle.HIGHEST_PROTOCOL)