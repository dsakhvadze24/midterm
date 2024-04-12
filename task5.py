import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re

# 1. Load the data
data = pd.read_csv("spam-data.csv")

# 2. Split data into features and labels
X = data['email_text']
y = data['label']

# 3. Build the logistic regression model and train it
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Locate emails.txt and parse it to extract features
with open("emails.txt", "r") as file:
    emails = file.readlines()

for email in emails:
    # Preprocess the email text
    email = re.sub(r'\W', ' ', email)
    email = re.sub(r'\s+', ' ', email)
    email = email.lower().strip()

    # Vectorize the email
    email_vectorized = vectorizer.transform([email])

    # Predict if the email is spam or not
    prediction = model.predict(email_vectorized)

    # Print the result
    if prediction[0] == 1:
        print("SPAM")
    else:
        print("NOT SPAM")

# 5. Analysis of the spam-data.csv file
# Extract feature importance from the model
feature_importance = pd.DataFrame(model.coef_[0], index=vectorizer.get_feature_names(), columns=['importance'])
feature_importance_sorted = feature_importance.sort_values(by='importance', ascending=False)

# Print top and bottom 10 important features
print("Top 10 important features:")
print(feature_importance_sorted.head(10))
print("\nBottom 10 important features:")
print(feature_importance_sorted.tail(10))

    
print('task is successful')