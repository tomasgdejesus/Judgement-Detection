import sklearn as skl
import lightgbm as lgb
import pandas as pd

upper_limit = 1e6
lower_limit = -1e6

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

def baseclassifier(train_data, test_data, dataset, mlmodel):
    model = get_mlmodel(mlmodel)
    le = skl.preprocessing.LabelEncoder()
    scaler = skl.preprocessing.StandardScaler()
    y = le.fit_transform(train_data['label'])
    train_data = train_data.drop('label', axis=1)
    train_data = train_data.clip(lower=lower_limit, upper=upper_limit)
    x = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    ytest = le.fit_transform(test_data['label'])
    test_data = test_data.drop('label', axis=1)
    test_data = test_data.clip(lower=lower_limit, upper=upper_limit)
    xtest = pd.DataFrame(scaler.fit_transform(test_data), columns=test_data.columns)
    model.fit(x, y)
    ypred = model.predict(xtest)

    accuracy = skl.metrics.accuracy_score(ytest, ypred)
    f1_score = skl.metrics.f1_score(ytest, ypred)
    print(f"Accuracy for {mlmodel} for {dataset}: {accuracy}")
    print(f"F1 score for {mlmodel} for {dataset}: {f1_score}")
    print(f"Report for {dataset}: {skl.metrics.classification_report(ytest, ypred)}")

    with open('metrics.txt', 'a') as f:
        f.write(f"Accuracy for {mlmodel} for {dataset}: {accuracy}\n")
        f.write(f"F1 score for {mlmodel} for {dataset}: {f1_score}\n")
        f.write(f"Report for {dataset} for {mlmodel}: {skl.metrics.classification_report(ytest, ypred)}\n")

def getreport(train_data, test_data, dataset):
    baseclassifier(train_data, test_data, dataset, 'Logistic Regression')
    baseclassifier(train_data, test_data, dataset, 'Random Forest')