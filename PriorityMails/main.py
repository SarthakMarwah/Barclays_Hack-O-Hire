
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import re
import datetime

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the CSV data into a DataFrame (assuming 'emails.csv' is your CSV file)
df = pd.read_csv('priority_dataset.csv')

# Combine subject and body text into a single feature 'content'
df['content'] = df['subject'] + ' ' + df['text']

# Define urgency keywords in descending order of priority and their frequency
urgency_keywords = [
    ('urgent', 20), ('important', 18), ('deadline', 16), ('emergency', 15),
    ('priority', 14), ('action required', 13), ('asap', 12), ('time-sensitive', 11),
    ('expedite', 10), ('rush', 9), ('critical situation', 8), ('high priority', 7),
    ('pressing', 6), ('top priority', 5), ('acute', 4), ('exigent', 3), ('impending', 2),
    ('crucial', 1), ('compulsory', 1), ('imperative', 1), ('vital', 1), ('serious', 1),
    ('mandatory', 1), ('essential', 1), ('immediate', 1), ('swift action', 1), 
    ('timely action', 1), ('prompt action', 1), ('urgent notification', 1),
    ('time-critical', 1), ('rapid response', 1), ('immediate attention', 1),
    ('swift response', 1), ('urgent attention', 1), ('rapid resolution', 1),
    ('swift resolution', 1), ('immediate response', 1), ('fast response', 1),
    ('quick response', 1), ('rapid attention', 1), ('immediate notification', 1),
    ('priority notification', 1), ('critical need', 1), ('immediate action required', 1),
    ('immediate assistance', 1), ('urgent action required', 1), ('priority task', 1),
    ('emergency situation', 1), ('urgent matter', 1), ('crucial matter', 1),
    ('time-critical task', 1), ('critical task', 1), ('urgent issue', 1),
    ('critical issue', 1), ('urgent requirement', 1), ('critical requirement', 1),
    ('urgent request', 1), ('critical request', 1)
]

# Extract date from the email body using regex
def extract_date(text):
    # Example regex pattern assuming date is in various formats
    pattern = r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b'  # Handles various date formats
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None

# Function to calculate priority based on urgency keywords and date proximity
def calculate_priority(row):
    priority = 0
    
    # Criteria 1: Check for urgency keywords and assign priority based on their presence and frequency
    for keyword, weight in urgency_keywords:
        if keyword in row['content'].lower():
            priority += weight
            break  # Exit loop once the highest priority keyword is found
    
    # Criteria 2: Check if date is present and compare it with current date
    if row['date']:
        current_date = datetime.date.today()
        email_date = datetime.datetime.strptime(row['date'], '%d/%m/%Y').date()  # Assuming date format as DD/MM/YYYY
        days_difference = (email_date - current_date).days
        priority += max(0, 10 - days_difference)  # Increase priority for emails closer to today
    
    # Criteria 3: Increase priority if email content contains specific keywords or phrases
    specific_keywords = ['reminder', 'follow-up', 'request', 'meeting', 'urgent', 'deadline']
    for keyword in specific_keywords:
        if keyword in row['content'].lower():
            priority += 5
            break  # Exit loop once a specific keyword is found
    
    return priority

# Apply date extraction and priority calculation functions to each row in the DataFrame
df['date'] = df['content'].apply(extract_date)
df['priority'] = df.apply(calculate_priority, axis=1)

# Sort emails by priority in descending order
df_sorted = df.sort_values(by='priority', ascending=False)

# Save sorted emails to 'mailsOutput.txt'
df_sorted.to_csv('mailsOutput.txt', sep='\t', index=False)
