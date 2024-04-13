import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('Train_db.csv')

# Preprocessing: Join text columns
data['combined_text'] = data['pre_text'] + ' ' + data['post_text']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['combined_text'], data['industry'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Encode the target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train_encoded)

# Predict on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(" ")
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print("F1 Score:", f1)
print(" ")

# Classification report
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print("Classification Report:")
print(report)

# Save the trained model and vectorizer
joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("Model training, evaluation, and saving completed.")
