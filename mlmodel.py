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
    return mlr, mrf, scaler

def predict_dataset(train_data, model_lr, model_rf, scalar, i, taskType, feature_data, isAugmentedFeature):
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

                if isAugmentedFeature:
                    prompt = exm["prompt"]
                    response = exm["response"]
                    df_X = pd.DataFrame([[helpfulness, correctness, coherence, complexity, verbosity, prompt, response]],
                    columns=["helpfulness", "correctness", "coherence", "complexity", "verbosity", "prompt", "response"])
                    
                    # Merge with features
                    temp = df_X.merge(feature_data, on=["prompt", "response"])
                    
                    # Drop extra columns
                    temp = temp.drop(columns=["prompt", "response", "group_id"], errors='ignore')
                    
                    X = temp.values
                    
                else:
                    # Construct X using the named variables
                    X = np.array([[helpfulness, correctness, coherence, complexity, verbosity]])
                if X.size:
                    X = scalar.transform(X)
                    value = model_lr.predict_proba(X)[:, 1][0]
                    y_grp_lr.append(logit(value))
                    value = model_rf.predict_proba(X)[:, 1][0]
                    y_grp_rf.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'helpsteer2', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'helpsteer2', taskType)
        return acc_lr, acc_rf, f1_lr, f1_rf
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
                if isAugmentedFeature:
                    response1 = exm["response1"]
                    response2 = exm["response2"]
                    df_X = pd.DataFrame([[score, response1, response2]],
                    columns=["score", "response1", "response2"])

                    # Merge with features
                    temp = df_X.merge(feature_data, on=["response1", "response2"])

                    # Drop extra columns
                    temp = temp.drop(columns=["response1", "response2", "group_id"], errors='ignore')
                    X = temp.values
                else:
                    # Construct X using the named variables
                    X = np.array([[score]])
                if X.size:
                    X = scalar.transform(X)
                    value = model_lr.predict_proba(X)[:, 1][0]
                    y_grp_lr.append(logit(value))
                    value = model_rf.predict_proba(X)[:, 1][0]
                    y_grp_rf.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'helpsteer3', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'helpsteer3', taskType)
        return acc_lr, acc_rf, f1_lr, f1_rf
    elif i == 2:
        feature_data = feature_data.drop(columns=["label", "ranking"], axis=1)
        ytest = (train_data['label']).map(label_map)
        ypred_lr = []
        ypred_rf = []
        for row in train_data.to_dict("records"):
            exm_list = row['examples']
            y_grp_lr = []
            y_grp_rf = []
            for exm in exm_list:
                ranking = exm["ranking"]
                if isAugmentedFeature:
                    query = exm["query"]
                    docs = exm["docs"]
                    df_X = pd.DataFrame([[ranking, query, docs]],
                    columns=["ranking", "query", "docs"])

                    # Merge with features
                    df_X["docs"] = df_X["docs"].apply(tuple)
                    temp = df_X.merge(feature_data, on=["query", "docs"])
                    temp['ranking'] = temp['ranking'].apply(d.normalize_ranking)
                    temp = d.expand_ranking_column(temp, "ranking")
                    # Drop extra columns
                    temp = temp.drop(columns=["query", "docs", "group_id"], errors='ignore')
                    X = temp.values
                else:
                    X = np.array([ranking])
                    X = np.array([d.normalize_ranking(row) for row in X])
                if X.size:
                    X = scalar.transform(X)
                    value = model_lr.predict_proba(X)[:, 1][0]
                    y_grp_lr.append(logit(value))
                    value = model_rf.predict_proba(X)[:, 1][0]
                    y_grp_rf.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'antique', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'antique', taskType)
        return acc_lr, acc_rf, f1_lr, f1_rf
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

                if isAugmentedFeature:
                    content = exm["content"]
                    df_X = pd.DataFrame([[rating, confidence, soundness, presentation, contribution, content]],
                    columns=["rating", "confidence", "soundness", "presentation", "contribution", "content"])

                    # Merge with features
                    df_X["content"] = df_X["content"].apply(d.clean_text)
                    temp = df_X.merge(feature_data, on=["content"])
                    # Drop extra columns
                    temp = temp.drop(columns=["content", "group_id"], errors='ignore')
                    X = temp.values
                else:
                    # Construct X using the named variables
                    X = np.array([[rating, confidence, soundness, presentation, contribution]])
                if X.size:
                    X = scalar.transform(X)
                    value = model_lr.predict_proba(X)[:, 1][0]
                    y_grp_lr.append(logit(value))
                    value = model_rf.predict_proba(X)[:, 1][0]
                    y_grp_rf.append(logit(value))
            ypred_lr.append(1 if np.sum(y_grp_lr) > 0 else 0)
            ypred_rf.append(1 if np.sum(y_grp_rf) > 0 else 0)
        acc_lr, f1_lr = display_metrics(ytest, ypred_lr, LOGISTIC_REGRESSION, 'neurips', taskType)
        acc_rf, f1_rf = display_metrics(ytest, ypred_rf, RANDOM_FOREST, 'neurips', taskType)
        return acc_lr, acc_rf, f1_lr, f1_rf

def predict(test_datasets, models_lr, models_rf, scaler, taskType, feature_data, isAugmentedFeature):
    results = []
    for i in range(len(models_lr)):
        metrics = predict_dataset(test_datasets[i], models_lr[i], models_rf[i], scaler[i], i, taskType, feature_data[i], isAugmentedFeature)
        results.append(metrics)
    return results