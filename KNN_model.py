import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # Step 1: Load the data (following the same pattern as existing models)
    data = pd.read_csv('data/enron_spam_data.csv')
    print("CSV loaded successfully.")
    print("Number of rows:", len(data))

    # Step 2: Combine subject and message text (same preprocessing as PAC_model.py)
    data['text'] = data['Subject'].fillna('') + ' ' + data['Message'].fillna('')
    X = data['text']
    y = data['Spam/Ham']

    # Step 3: Split dataset (using same test_size as PAC_model.py for consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Step 4: Text vectorization using TF-IDF (same parameters as existing models)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, lowercase=False)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("TF-IDF vectorization completed.")

    # Step 5: Train k-Nearest Neighbors model
    print("Training k-NN model...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_tfidf, y_train)

    # Step 6: Make predictions
    print("Making predictions...")
    y_pred = knn_model.predict(X_test_tfidf)

    # Step 7: Evaluate the model
    print("\n" + "="*50)
    print("k-NN Model Evaluation Results")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("="*50)


if __name__ == '__main__':
    main()

