import numpy as np
import math
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay

mean_fpr = np.linspace(0, 1, 100)

def clean_data(sacc_file, BATCH_SUBJECTS):
    if BATCH_SUBJECTS:
        METADATA = ['trial', 'item', 'sn']
    else:
        METADATA = ['trial', 'item']

    data_df = sacc_file.drop(METADATA, axis=1)

    # add log transformed gaze durations
    data_df['gz'] = data_df['gaze'].map(lambda x: math.log(x) if x >= 1 else 0)
    data_df = data_df.drop("gaze", axis=1)

    # convert group labels to 0 (control) and 1 (dyslexic)
    data_df['group'] = data_df['group'].map(lambda x: int(x + 0.5))

    return data_df


def aggregate_data(data_df, BATCH_SUBJECTS):
    # group data by subject
    if BATCH_SUBJECTS:
        grouped_data = data_df.groupby('subj')
    else:
        grouped_data = data_df.groupby(['subj', 'sn'])

    # calculate means and standard deviations of all ET features
    means = grouped_data.mean()
    stds = grouped_data.std()

    # join the means and standard deviations in one dataframe
    feature_group_df = means.join(stds, lsuffix="_mean", rsuffix="_std")

    return feature_group_df


def dataframe_to_vecs(data_df, folds, BATCH_SUBJECTS):
    feature_folds = []
    label_folds = []
    # get subj column
    if BATCH_SUBJECTS:
        data_df['subj'] = data_df.index
    else:
        all_subj_sent = data_df.index.to_numpy()
        data_df['subj'] = [subj_sent[0] for subj_sent in all_subj_sent]
    for _, fold in enumerate(folds):
        fold_data = data_df[data_df['subj'].isin(fold)]
        feature_folds.append(
            fold_data.drop(['group_mean', 'group_std', 'subj'], axis=1).to_numpy()
        )
        label_folds.append(fold_data[['group_mean']].to_numpy().transpose().flatten()) 
    return feature_folds, label_folds


def get_train_dev_test_splits(n_folds, i_test, j_dev, feature_splits, label_splits):  
    counter = 0
    test_features = feature_splits[i_test]
    test_labels = label_splits[i_test]

    dev_features = feature_splits[j_dev]
    dev_labels = label_splits[j_dev]

    train_labels = np.array([])
    for idx, elem in enumerate(feature_splits):
        if idx != i_test and idx != j_dev:
            if counter == 0:
                train_features = elem
                counter += 1
            else:
                train_features = np.concatenate((train_features, elem), axis=0)

    for idx, elem in enumerate(label_splits):
        if idx != i_test and idx != j_dev:
            train_labels = np.concatenate((train_labels, elem), axis=0)

    return train_features, dev_features, test_features, train_labels, dev_labels, test_labels


def svm_model(train_features, train_labels, param):
    classifier = svm.SVC(kernel='linear', C=param, probability=False)
    classifier.fit(train_features, train_labels)
    return classifier


def rfe_model(train_features, train_labels, param, n_features):
    # define the classifier model
    classifier = svm.SVC(kernel='linear', C=param, probability=False)

    # recursive feature selection
    selector = RFE(classifier, n_features_to_select=n_features, step=1)
    selector = selector.fit(train_features, train_labels)

    return selector


def evaluate(classifier, features, labels):
    pred = classifier.predict(features)
    accuracy = accuracy_score(labels, pred)
    # specificity = tn / (tn + fp)
    # specificity = specificity_score(labels, pred)
    # recall: tp / (tp + fn)
    recall = recall_score(labels, pred, zero_division=np.nan) 
    # presicion = tp / (tp + fp) 
    precision = precision_score(labels, pred, zero_division=np.nan) 
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(labels, pred, zero_division=np.nan)  

    return [accuracy, precision, recall, f1], pred # return pred here


def compare_hyperparameters(train_features, train_labels, dev_features, dev_labels, c_params, n_features): 
    # array that collect the scores for each param and each number of features
    eval_scores_all_params = np.zeros([len(c_params), n_features, 4])
    
    # for each parameter and each number of feature train a classifier
    for param in c_params:
        for feature in range(n_features, 0, -1):
            rfe = rfe_model(train_features, train_labels, param, feature)
            # evaluate the classifier on the dev set
            scores_per_feature, _ = evaluate(rfe, dev_features, dev_labels)
            # include the scores in the score array
            eval_scores_all_params[c_params.index(param), feature-1] += scores_per_feature
        
    return np.array(eval_scores_all_params)


def nested_cv_rfe(feature_folds, label_folds, n_folds, c_params, n_features, ax):
    # array to collect the scores during training
    saving_scores = {}
    scores = np.zeros([n_folds, len(c_params), n_features, 4])
    scores_per_feature = np.zeros([n_folds, n_features, 4])
    # lists to collect the best params and the best number of features per fold and the final evaluations per fold
    best_params = []
    best_n_features = []
    CV_eval = []

    # set index for test set
    for i_test in range(n_folds):
        # set index for dev set
        for j_dev in range(n_folds):
            # try each combination of test and dev sets
            if i_test != j_dev:
                train_features, dev_features, test_features, train_labels, dev_labels, test_labels = get_train_dev_test_splits(
                    n_folds, i_test, j_dev, feature_folds, label_folds)
                scaler = MinMaxScaler()
                train_features = scaler.fit_transform(train_features)
                dev_features = scaler.transform(dev_features)
                # compare the 8 hyperparameters on each test - dev combination
                scores[i_test] += compare_hyperparameters(train_features, train_labels, dev_features, dev_labels,
                                                          c_params, n_features)

        # calculate the mean of evaluations for each parameter over all n_folds-1 dev sets
        # (-1, because one fold is always excluded as test set)
        scores[i_test] /= (n_folds - 1)

        # for each fold, extract the index of the best parameter and best number of features regarding f1-score
        f1 = scores[i_test, :, :, 3]
        # returns a tuple with indices (row, column) for the max value: row corresponds to param, column to n_features
        indices = np.unravel_index(np.argmax(f1), f1.shape)
        # retrieve best parameter from list via its index
        best_param = c_params[indices[0]]
        best_params.append(best_param)
        # retrieve best number of features by adding 1 to its index (since index starts at 0)
        best_n_feature = indices[1] + 1
        best_n_features.append(best_n_feature)

        # train RFE model using best hyperparameter from inner cross-validation, on 90% train data (no dev set)
        train_features_fold, dev_features_fold, test_features_fold, train_labels_fold, dev_labels_fold, test_labels_fold = get_train_dev_test_splits(
            n_folds, i_test, i_test, feature_folds, label_folds)
        scaler = MinMaxScaler()
        train_features_fold = scaler.fit_transform(train_features_fold)
        test_features_fold = scaler.transform(test_features_fold)

        final_model = rfe_model(train_features_fold, train_labels_fold, best_param, best_n_feature)
        test_scores, pred = evaluate(final_model, test_features_fold, test_labels_fold)
        
        tprs = []
        aucs = []
        viz = RocCurveDisplay.from_estimator(
            final_model,
            test_features_fold,
            test_labels_fold,
            name="ROC fold {}".format(i_test + 1),
            alpha=0.3,
            lw=1,
            ax=ax,
            drop_intermediate=False
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        saving_scores[i_test] = [test_labels_fold, pred]

        CV_eval += [test_scores]

    # calcualte the mean evaluation scores over all n_folds
    final_scores_mean = np.mean(CV_eval, axis=0)
    final_scores_std = np.std(CV_eval, axis=0)
    print('final scores mean :', final_scores_mean)
    print('final scores sd: ', final_scores_std)
    return final_scores_mean, final_scores_std, best_params, best_n_features, CV_eval, saving_scores, aucs
