import csv
import sklearn as skl
import lightgbm as lgb
import pandas as pd
import numpy as np
from scipy.special import logit
import data as d

import warnings
warnings.filterwarnings("ignore")

upper_limit = 1e6
lower_limit = -1e6
LOGISTIC_REGRESSION = 'Logistic Regression'
RANDOM_FOREST = 'Random Forest'
LIGHT_GBM = 'Light gbm'
label_map = {'Human': 0, 'LLM': 1}

def display_metrics(ytest, ypred, mlmodel, dataset, taskType):
    accuracy = skl.metrics.accuracy_score(ytest, ypred)
    f1_score = skl.metrics.f1_score(ytest, ypred)
    print(f"Accuracy for {mlmodel} for {dataset}: {accuracy}")
    print(f"F1 score for {mlmodel} for {dataset}: {f1_score}")
    print(f"Report for {dataset}: {skl.metrics.classification_report(ytest, ypred)}")

    with open('metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([taskType, mlmodel, dataset, accuracy, f1_score])
    return accuracy, f1_score

def get_mlmodel(type):
    model = ""
    if type == LOGISTIC_REGRESSION:
        model = skl.linear_model.LogisticRegression()
    elif type == RANDOM_FOREST:
        model = skl.ensemble.RandomForestClassifier(
            n_estimators=100,  # number of trees
            max_depth=None,    # maximum depth of trees
            random_state=42
        )
    elif type == LIGHT_GBM:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=-1,   # -1 means no limit
            random_state=42
        )
    return model

def baseclassifier(train_data, test_data, dataset, mlmodel, taskType):
    model = get_mlmodel(mlmodel)
    scaler = skl.preprocessing.StandardScaler()
    y = train_data['label'].map(label_map)
    train_data = train_data.drop('label', axis=1)
    train_data = train_data.clip(lower=lower_limit, upper=upper_limit)
    x = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    ytest = test_data['label'].map(label_map)
    test_data = test_data.drop('label', axis=1)
    test_data = test_data.clip(lower=lower_limit, upper=upper_limit)
    xtest = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
    model.fit(x, y)
    ypred = model.predict(xtest)

    display_metrics(ytest, ypred, mlmodel, dataset, taskType)

    return model, scaler

def getreport(train_data, test_data, dataset, taskType):
    mlr, scaler = baseclassifier(train_data, test_data, dataset, LOGISTIC_REGRESSION, taskType)
    mrf, scaler = baseclassifier(train_data, test_data, dataset, RANDOM_FOREST, taskType)
    mlg, scaler = baseclassifier(train_data, test_data, dataset, LIGHT_GBM, taskType)
    return mlr, mrf, mlg, scaler

def predict_dataset(train_data, model_lr, model_rf, model_lg, scalar, i, taskType):
    if i == 0:
        ytest = train_data['label'].map(label_map)
        ypred_lr = []
        ypred_rf = []
        ypred_lg = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            y_grp_lg = []
            for exm in exm_list:
                # Adding column names explicitly
                helpfulness = exm["helpfulness"]
                correctness = exm["correctness"]
                coherence = exm["coherence"]
                complexity = exm["complexity"]
                verbosity = exm["verbosity"]

                # Construct X using the named variables
                X = np.array([[helpfulness, correctness, coherence, complexity, verbosity]])
                X = scalar.transform(X)
                value = model_lr.predict_proba(X)[:, 1][0]
                y_grp_lr.append(logit(value))
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(logit(value))
                value = model_lg.predict_proba(X)[:, 1][0]
                y_grp_lg.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
            ypred_lg.append(1 if np.sum(y_grp_lr) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'helpsteer2', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'helpsteer2', taskType)
        acc_lg, f1_lg = display_metrics(ytest, ypred_lr, LIGHT_GBM, 'helpsteer2', taskType)
        return acc_lr, acc_rf, acc_lg, f1_lr, f1_rf, f1_lg
    elif i == 1:
        ytest = (train_data['label']).map(label_map)
        ypred_lr = []
        ypred_rf = []
        ypred_lg = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            y_grp_lg = []
            for exm in exm_list:
                # Adding column names explicitly
                score = exm["score"]

                # Construct X using the named variables
                X = np.array([[score]])
                X = scalar.transform(X)
                value = model_lr.predict_proba(X)[:, 1][0]
                y_grp_lr.append(logit(value))
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(logit(value))
                value = model_lg.predict_proba(X)[:, 1][0]
                y_grp_lg.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
            ypred_lg.append(1 if np.sum(y_grp_lr) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'helpsteer3', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'helpsteer3', taskType)
        acc_lg, f1_lg = display_metrics(ytest, ypred_lr, LIGHT_GBM, 'helpsteer3', taskType)
        return acc_lr, acc_rf, acc_lg, f1_lr, f1_rf, f1_lg
    elif i == 2:
        ytest = (train_data['label']).map(label_map)
        ypred_lr = []
        ypred_rf = []
        ypred_lg = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            y_grp_lg = []
            for exm in exm_list:
                X = np.array([  
                    exm["ranking"]
                ])
                X = np.array([d.normalize_ranking(row) for row in X])
                X = scalar.transform(X)
                value = model_lr.predict_proba(X)[:, 1][0]
                y_grp_lr.append(logit(value))
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(logit(value))
                value = model_lg.predict_proba(X)[:, 1][0]
                y_grp_lg.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
            ypred_lg.append(1 if np.sum(y_grp_lr) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'antique', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'antique', taskType)
        acc_lg, f1_lg = display_metrics(ytest, ypred_lr, LIGHT_GBM, 'antique', taskType)
        return acc_lr, acc_rf, acc_lg, f1_lr, f1_rf, f1_lg
    elif i == 3:
        ytest = train_data['label'].map(label_map)
        ypred_lr = []
        ypred_rf = []
        ypred_lg = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            y_grp_lg = []
            for exm in exm_list:
                # Adding column names explicitly
                rating = exm["rating"]
                confidence = exm["confidence"]
                soundness = exm["soundness"]
                presentation = exm["presentation"]
                contribution = exm["contribution"]

                # Construct X using the named variables
                X = np.array([[rating, confidence, soundness, presentation, contribution]])
                X = scalar.transform(X)
                value = model_lr.predict_proba(X)[:, 1][0]
                y_grp_lr.append(logit(value))
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(logit(value))
                value = model_lg.predict_proba(X)[:, 1][0]
                y_grp_lg.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
            ypred_lg.append(1 if np.sum(y_grp_lr) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'neurips', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'neurips', taskType)
        acc_lg, f1_lg = display_metrics(ytest, ypred_lr, LIGHT_GBM, 'neurips', taskType)
        return acc_lr, acc_rf, acc_lg, f1_lr, f1_rf, f1_lg

def predict(test_datasets, models_lr, models_rf, modelslg, scaler, taskType):
    results = []
    for i in range(len(models_lr)):
        metrics = predict_dataset(test_datasets[i], models_lr[i], models_rf[i], modelslg[i], scaler[i], i, taskType)
        results.append(metrics)
    return results