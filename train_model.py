import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Get the exact folder where THIS script is saved
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'spam_model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
csv_path = os.path.join(current_dir, 'mail_data.csv')

print(f"Checking for CSV at: {csv_path}")

if not os.path.exists(csv_path):
    print("❌ ERROR: mail_data.csv not found in this folder!")
else:
    # 1. Load Data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    mail_data = df.where((pd.notnull(df)), '')
    
    # Label Encoding
    mail_data['category'] = mail_data['category'].map({'spam': 0, 'ham': 1})
    mail_data['category'] = mail_data['category'].fillna(0).astype(int)

    X = mail_data['message']
    Y = mail_data['category']

    # 2. Train Model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    # 3. Save files using Absolute Paths
    pickle.dump(model, open(model_path, 'wb'))
    pickle.dump(feature_extraction, open(vectorizer_path, 'wb'))

    print(f"✅ SUCCESS! Created:\n1. {model_path}\n2. {vectorizer_path}")