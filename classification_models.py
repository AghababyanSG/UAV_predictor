import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


dataset_path = 'archive/data_2m.csv'
data = pd.read_csv(dataset_path)

replacement_map = {2.0: 0, 3.0: 1, 4.0: 2}
data['type'] = data['type'].replace(replacement_map)

print("Data has been read")

try:
    X = data.drop('type', axis=1)
    y = data['type']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    models = [
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        SGDClassifier(max_iter=1000, tol=0.01, n_jobs=-1),
        XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1,
                      objective='binary:logistic', n_jobs=-1),
        DummyClassifier(strategy="most_frequent")
    ]
    models_names = ['RandomForestClassifier',
                    'SGDClassifier', 'XGBClassifier', 'DummyClassifier']

    i = 0
    for model in models:
        try:
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            print(f"\n{models_names[i]}")
            print(f"Accuracy score on validation set: {acc}")

            y_test_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)
            print(f"Accuracy score on test set: {acc}")

        except Exception as e:
            print(f"ERROR: {e}")
        i += 1


except:

    print(0)
    dataset_path = 'archive/data_2m.csv'
    data = pd.read_csv(dataset_path)
    X = data.drop('type', axis=1)
    y = data['type']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    models = [RandomForestClassifier(n_estimators=100, random_state=42),
                 SGDClassifier(max_iter=1000, tol=0.01),
                 XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic'),
                 DummyClassifier(strategy="most_frequent")]
    


    models_names = ['RandomForestClassifier',
                    'SGDClassifier', 'XGBClassifier', 'DummyClassifier']

    i = 0
    for model in models:
        try:
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_val_pred)
            print(f"\n{models_names[i]}")
            print(f"Accuracy score on validation set: {acc}")

            y_test_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_test_pred)

            print(f"Accuracy score on test set: {acc}")

        except:
            print("ERROR")
        i += 1
