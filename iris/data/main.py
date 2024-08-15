import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = pd.read_csv('IRIS_train.csv')
test_data = pd.read_csv('IRIS_test.csv')

X_train = train_data.drop(columns=['species'])
y_train = train_data['species']

X_test = test_data.drop(columns=['species'])
y_test = test_data['species']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifier = RandomForestClassifier(random_state=1)
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(report)
print('\nConfusion Matrix:')
print(conf_matrix)