import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the spam-data.csv file
data = pd.read_csv('spam-data.csv')

# Explore the data
print(data.head())
print(data.describe())
print(data.columns)  # Check the column names

# Prepare the data
X = data['Text'].values  # Assuming the email text is in the 'Text' column
y = data['Prediction'].values  # Assuming the spam/not spam label is in the 'Prediction' column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X_train.reshape(-1, 1), y_train)

# Test the first email from the emails.txt file
with open('emails.txt', 'r') as file:
    first_email = file.readline().strip()

prediction = model.predict([first_email])
print(f"The first email is {'spam' if prediction[0] else 'not spam'}.")


print('task is over successfully')