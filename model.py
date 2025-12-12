import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Load the dataset
df = pd.read_csv("student_scores.csv")

# 2. Separate input (X) and output (y)
X = df[["Hours"]]     # Features must be 2D
y = df["Score"]       # Target is 1D

# 3. Create a model and train it
model = LinearRegression()
model.fit(X, y)

# 4. Test the model with an example
hours = [[6]]   # A student who studies 6 hours
predicted_score = model.predict(hours)

print("Predicted Score for 6 hours:", predicted_score[0])
