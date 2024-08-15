import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'IRIS.csv'
data = pd.read_csv(file_path)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('IRIS_train.csv', index=False)
test_data.to_csv('IRIS_test.csv', index=False)

print("Train and test datasets have been created and saved as 'IRIS_train.csv' and 'IRIS_test.csv'.")