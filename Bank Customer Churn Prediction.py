# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Display all columns
pd.set_option('display.max_columns', None)

# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Data preprocessing
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical features (Geography and Gender)
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Split the data into features (X) and the target variable (y)
X = data.drop('Exited', axis=1)
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

color = sns.color_palette(["lightblue", "blue"])

# Display the confusion matrix as a visual image
sns.heatmap(confusion, annot=True, fmt='d', cmap=color)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\n------------------------------------------------------")
print("------------------------------------------------------")
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("\n------------------------------------------------------")
print("------------------------------------------------------")

print("Classification Report:\n", report)
