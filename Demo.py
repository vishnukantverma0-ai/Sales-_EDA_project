import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Create a dummy dataset (In a real project, you'd load a CSV file)
data = {
    'sq_feet': [750, 800, 1100, 1300, 1500, 1800, 2200, 2500],
    'bedrooms': [1, 1, 2, 2, 3, 3, 4, 4],
    'price': [200000, 210000, 250000, 280000, 310000, 350000, 420000, 480000]
}
df = pd.DataFrame(data)

# 2. Split data into Features (X) and Target (y)
X = df[['sq_feet', 'bedrooms']] # What we use to predict
y = df['price']                # What we want to predict

# 3. Split into Training and Testing sets
# We keep 20% of data hidden so we can test the model later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make a prediction! 
# Let's predict the price of a 2000 sq ft house with 3 bedrooms
new_house = np.array([[2000, 3]])
prediction = model.predict(new_house)

print(f"Predicted Price: ${prediction[0]:,.2f}")