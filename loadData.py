import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/enron_spam_data.csv')
print(data.head())

X = data[['Subject', 'Message']];
y = data['Spam/Ham'];

# 70 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100);

print('X_train : ')
print(X_train.head())
print('')
print('X_test : ')
print(X_test.head())
print('')
print('y_train : ')
print(y_train.head())
print('')
print('y_test : ')
print(y_test.head())