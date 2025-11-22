from datetime import datetime
import os
import json

import sklearn as skl
import lightgbm as lgb
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

upper_limit = 1e6
lower_limit = -1e6
LOGISTIC_REGRESSION = 'Logistic Regression'
RANDOM_FOREST = 'Random Forest'
label_map = {'Human': 0, 'LLM': 1}

def display_metrics(ytest, ypred, mlmodel, dataset, run_metadata=None):
    accuracy = skl.metrics.accuracy_score(ytest, ypred)
    f1_score = skl.metrics.f1_score(ytest, ypred)
    print(f"Accuracy for {mlmodel} for {dataset}: {accuracy}")
    print(f"F1 score for {mlmodel} for {dataset}: {f1_score}")
    print(f"Report for {dataset}: {skl.metrics.classification_report(ytest, ypred)}")

    # metrics.txt legacy
    with open('metrics.txt', 'a') as f:
        f.write(f"Accuracy for {mlmodel} for {dataset}: {accuracy}\n")
        f.write(f"F1 score for {mlmodel} for {dataset}: {f1_score}\n")
        f.write(f"Report for {dataset} for {mlmodel}: {skl.metrics.classification_report(ytest, ypred)}\n")

    # save to json
    if not isinstance(run_metadata, dict):
        run_metadata = {} # just in case

    run_record = {
        'timestamp': datetime.utcnow().isoformat(),
        'accuracy': accuracy,
        'f1_score': f1_score,
        'mlmodel': mlmodel,
        'dataset': dataset,
        'classification_report': skl.metrics.classification_report(ytest, ypred, output_dict=True),
        'run_metadata': run_metadata
    }

    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    else:
        metrics = []

    metrics.append(run_record)
    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

def get_mlmodel(type):
    model = ""
    if type == 'Logistic Regression':
        model = skl.linear_model.LogisticRegression()
    elif type == 'Random Forest':
        model = skl.ensemble.RandomForestClassifier(
            n_estimators=100,  # number of trees
            max_depth=None,    # maximum depth of trees
            random_state=42
        )
    return model

def baseclassifier(train_data, test_data, dataset, mlmodel, run_metadata=None):
    model = get_mlmodel(mlmodel)
    scaler = skl.preprocessing.StandardScaler()
    y = train_data['label'].map(label_map)
    train_data = train_data.drop('label', axis=1)
    train_data = train_data.clip(lower=lower_limit, upper=upper_limit)
    x = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    ytest = test_data['label'].map(label_map)
    test_data = test_data.drop('label', axis=1)
    test_data = test_data.clip(lower=lower_limit, upper=upper_limit)
    xtest = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)
    model.fit(x, y)
    ypred = model.predict(xtest)

    display_metrics(ytest, ypred, mlmodel, dataset, run_metadata)

    return model, scaler

def getreport(train_data, test_data, dataset, run_metadata=None):
    mlr, scaler = baseclassifier(train_data, test_data, dataset, LOGISTIC_REGRESSION, run_metadata)
    mrf, scaler = baseclassifier(train_data, test_data, dataset, RANDOM_FOREST, run_metadata)
    return mlr, mrf, scaler

def predict_dataset(train_data, model_lr, model_rf, scalar, i, run_metadata=None):
    if i == 0:
        ytest = train_data['label'].map(label_map)
        ypred_lr = []
        ypred_rf = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
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
                y_grp_lr.append(value)
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(value)
            ypred_lr.append(1 if np.mean(y_grp_lr) > 0.5 else 0)
            ypred_rf.append(1 if np.mean(y_grp_rf) > 0.5 else 0)

        display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'helpsteer2', run_metadata)
        display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'helpsteer2', run_metadata)
    
    elif i == 1:
        ytest = (train_data['label']).map(label_map)
        ypred_lr = []
        ypred_rf = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            for exm in exm_list:
                # Adding column names explicitly
                score = exm["score"]

                # Construct X using the named variables
                X = np.array([[score]])
                X = scalar.transform(X)
                value = model_lr.predict_proba(X)[:, 1][0]
                y_grp_lr.append(value)
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(value)
            value = np.mean(y_grp_lr)
            value = 1 if value > 0.5 else 0
            ypred_lr.append(value)
            value = np.mean(y_grp_rf)
            value = 1 if value > 0.5 else 0
            ypred_rf.append(value)
        display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'helpsteer3', run_metadata)
        display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'helpsteer3', run_metadata)
        
    elif i == 2:
        ytest = (train_data['label']).map(label_map)
        ypred_lr = []
        ypred_rf = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            for exm in exm_list:
                X = np.array([  
                    exm["ranking"]
                ])
                X = scalar.transform(X)
                value = model_lr.predict_proba(X)[:, 1][0]
                y_grp_lr.append(value)
                value = model_rf.predict_proba(X)[:, 1][0]
                y_grp_rf.append(value)
            value = np.mean(y_grp_lr)
            value = 1 if value > 0.5 else 0
            ypred_lr.append(value)
            value = np.mean(y_grp_rf)
            value = 1 if value > 0.5 else 0
            ypred_rf.append(value)
        display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'antique', run_metadata)
        display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'antique', run_metadata)

    elif i == 3:
        ytest = train_data['label'].map(label_map)
        ypred_lr = []
        ypred_rf = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
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
                value = model_lr.predict_proba(X)[:, 1]
                y_grp_lr.append(value)
                value = model_rf.predict_proba(X)[:, 1]
                y_grp_rf.append(value)
            value = np.mean(y_grp_lr)
            value = 1 if value > 0.5 else 0
            ypred_lr.append(value)
            value = np.mean(y_grp_rf)
            value = 1 if value > 0.5 else 0
            ypred_rf.append(value)
        display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'neurips', run_metadata)
        display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'neurips', run_metadata)


def predict(test_datasets, models_lr, models_rf, scaler, run_metadata=None):
    for i in range(len(models_lr)):
        predict_dataset(test_datasets[i], models_lr[i], models_rf[i], scaler[i], i, run_metadata)