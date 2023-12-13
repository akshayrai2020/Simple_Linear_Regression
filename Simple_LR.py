# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('Housing.csv')

# Explore the data
print(data.head())

# Visualize the data
plt.scatter(data['AREA'], data['PRICE'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Housing Price vs. Area')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['AREA']], data['PRICE'], test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Visualize the regression line
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Housing Price Prediction with Linear Regression')
plt.show()

# Save the model (optional)
# import joblib
# joblib.dump(model, 'linear_regression_model.joblib')

