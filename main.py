import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# STEP 1: Creating a dataset
# In a real job, this data would come from an SQL query (Oracle/PostgreSQL)
data = {
    'contract_months': [1, 12, 24, 2, 36, 4, 48, 6, 12, 24],
    'monthly_spend': [100, 500, 1000, 150, 2000, 200, 3000, 250, 550, 1100],
    'churn': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0] # 1 = customer left, 0 = customer stayed
}

df = pd.DataFrame(data)

# STEP 2: Features (X) and Target (y)
# X is what we use to predict, y is what we want to find out
X = df[['contract_months', 'monthly_spend']]
y = df['churn']

# STEP 3: Split the data into Training and Testing sets
# We use 80% to train the "brain" and 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Creating the AI Model (Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# STEP 5: Testing the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100}%")

# STEP 6: Predicting a new customer
# Let's check a customer with 3 months of contract and 180 spent
new_customer = [[3, 180]]
result = model.predict(new_customer)
print(f"Prediction for new customer (1=Churn, 0=Stay): {result[0]}")