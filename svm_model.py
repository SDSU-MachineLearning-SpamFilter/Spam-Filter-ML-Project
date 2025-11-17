import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# step 1: load the data (will take some time to process as it is huge)
print("Starting SVM script...")
data = pd.read_csv('data/enron_spam_data.csv')
print("CSV loaded successfully.")
print("Number of rows:", len(data))


# combine subject and message text
data['text'] = data['Subject'].fillna('') + ' ' + data['Message'].fillna('')
X = data['text']
y = data['Spam/Ham']

# step 2: split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# step 3: text vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=False)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# step 4: train SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_tfidf, y_train)

# step 5: predict
y_pred = svm_model.predict(X_test_tfidf)

# step 6: evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
