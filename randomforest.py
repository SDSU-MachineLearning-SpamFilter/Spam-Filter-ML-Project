import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def main():
    # Load and split the data
    data = pd.read_csv('data/enron_spam_data.csv')

    data['X'] = data['Subject'].fillna('') + ' ' + data['Message'].fillna('')
    X = data['X']
    y = data['Spam/Ham']

    # 70 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # TF-IDF Scaling
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=False)
    sX_train = vectorizer.fit_transform(X_train)
    sX_test = vectorizer.transform(X_test)

    # Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=500, random_state=42)
    classifier.fit(sX_train, y_train)
    predicted_y = classifier.predict(sX_test)

    # Evaluate the model
    print('Random Forest Evaluation results:', flush=True)
    print('Accuracy:', accuracy_score(y_test, predicted_y), flush=True)
    print('\nClassification Report:\n', classification_report(y_test, predicted_y), flush=True)
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, predicted_y), flush=True)


if __name__ == '__main__':
    main()