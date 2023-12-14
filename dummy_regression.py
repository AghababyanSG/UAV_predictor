import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = 'archive/data_2m.csv'
df = pd.read_csv(file_path)

# Splitting the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(df.drop('type', axis=1), df['type'], test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# List of classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate each classifier
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'{clf_name}: Accuracy on Validation Set: {accuracy}')

# Choose the best classifier based on validation accuracy
best_classifier_name = max(classifiers, key=lambda k: accuracy_score(y_val, classifiers[k].predict(X_val)))
best_classifier = classifiers[best_classifier_name]

# Evaluate the best classifier on the test set
y_test_pred = best_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'\nBest Classifier: {best_classifier_name} with Accuracy on Test Set: {test_accuracy}')

