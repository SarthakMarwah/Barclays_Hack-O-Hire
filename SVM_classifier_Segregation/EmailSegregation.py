import os
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the test dataset
test_data = pd.read_csv('Test_db.csv')

# Preprocessing: Join pre_text and post_text columns
test_data['combined_text'] = test_data['pre_text'] + ' ' + test_data['post_text']

# Load the trained model and vectorizer
svm_model = joblib.load('svm_model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# TF-IDF Vectorization for test data
X_test_tfidf = tfidf_vectorizer.transform(test_data['combined_text'])

# Predict on test data
test_predictions = svm_model.predict(X_test_tfidf)
predicted_labels = label_encoder.inverse_transform(test_predictions)

# Define dummy Gmail IDs for each category
category_emails  = {
    'Pharmaceutical': 'sh7758@srmist.edu.in',
    'Finance': 'sarthak.marwah@aiesec.net',
    'Energy': 'jaiswalaparna27@hotmail.com',
    'Travel': 'odetocode@098@gmail.com',
    'Technology': 'samarthagldev@gmail.com',
}

# Connect to SMTP server
smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = 'sarthakmarwah@gmail.com'  # Your Gmail username
smtp_password = 'jzmn vifm qcbh gfqn'  # Your Gmail password
smtp_sender = 'sarthakmarwah@gmail.com'  # Sender email address

server = smtplib.SMTP(smtp_server, smtp_port)
server.starttls()
server.login(smtp_username, smtp_password)


# Route emails based on predicted labels
for idx, label in enumerate(predicted_labels):
    email_content = test_data.iloc[idx]['pre_text'] + ' ' + test_data.iloc[idx]['post_text']
    recipient_email = category_emails.get(label)
    subject = f"Email for {label}"

    # Perform email routing or processing based on recipient_email
    if recipient_email:
        print(f"Email routed to {recipient_email} for category {label}")
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = smtp_sender
        msg['To'] = recipient_email # Use default if category not found in dummy_emails
        msg['Subject'] = subject
        
         # Attach email content
        msg.attach(MIMEText(email_content, 'plain'))

        # Send the email
        server.send_message(msg)

    else:
        print(f"No email address found for category {label}, email not sent")


# Disconnect from SMTP server
server.quit()

print("Emails sent to dummy Gmail IDs based on categories.")
