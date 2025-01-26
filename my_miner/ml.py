# Note :
# Achieving over 95% accuracy in predicting sales prices is straightforward with a 
# simple model. However, it is the time dependency aspect that significantly enhances 
# miners' emissions. For this reason, I have chosen not to share the sales
# date prediction script.

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# 1. Configuration & Constants
# =============================================================================

FILE_PATH = '/Users/maxbernheim/Desktop/ML_Stanford/DrH/TEST/nextplace_training_data_TEST_filled.csv'
MODEL_FILENAME = 'linear_regression_model.pkl'
ENCODERS_FILENAME = 'label_encoders.pkl'

# Features to be used in the regression model
SELECTED_FEATURES = [
    'beds', 'baths', 'sqft', 'lot_size', 'hoa_dues',
    'listing_price', 'longitude', 'property_type'
]

# =============================================================================
# 2. Load and Inspect Data
# =============================================================================

print('Loading data...')
data = pd.read_csv(FILE_PATH)
print(f'Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.')

# =============================================================================
# 3. Feature Validation & Preparation
# =============================================================================

# Check if all SELECTED_FEATURES are present
missing_features = [feat for feat in SELECTED_FEATURES if feat not in data.columns]
if missing_features:
    raise ValueError(f"The following selected features are not in the dataset: {missing_features}")

# Convert sale_price to numeric
data['sale_price'] = pd.to_numeric(data['sale_price'], errors='coerce')

# Keep records with sale_price >= 10,000
data = data[data['sale_price'] >= 10000].reset_index(drop=True)

# Separate numeric and categorical features
numeric_cols = data[SELECTED_FEATURES].select_dtypes(include=['float64', 'int64']).columns.tolist()
object_cols = data[SELECTED_FEATURES].select_dtypes(include=['object']).columns.tolist()

# Fill missing categorical values with 'N/A' (or another placeholder)
data[object_cols] = data[object_cols].fillna('N/A')

# =============================================================================
# 4. Encode Categorical Features
# =============================================================================

label_encoders = {}

for col in object_cols:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col].astype(str))
    label_encoders[col] = encoder

print('Finished encoding categorical features.')

# =============================================================================
# 5. Prepare Training and Test Sets
# =============================================================================

X = data[SELECTED_FEATURES]   # Feature matrix
y = data['sale_price']        # Target vector

# Split data (80% test, 20% train – though usually we do the opposite, 
# but preserving the original logic here)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=6
)

print(f'Training set size: {X_train.shape[0]} samples')
print(f'Testing set size: {X_test.shape[0]} samples')

# =============================================================================
# 6. Train the Model
# =============================================================================

print('Training the Linear Regression model...')
model = LinearRegression()
model.fit(X_train, y_train)
print('Model training complete.')

# =============================================================================
# 7. Save the Model and Encoders
# =============================================================================

joblib.dump(model, MODEL_FILENAME)
print(f'Model saved to {MODEL_FILENAME}')

joblib.dump(label_encoders, ENCODERS_FILENAME)
print(f'Label encoders saved to {ENCODERS_FILENAME}')

# =============================================================================
# 8. Evaluate Model Performance (Custom Metric)
# =============================================================================

def calculate_price_score(actual_price: float, predicted_price: float) -> float:
    """
    Computes a custom 'price score' based on the relative percentage difference
    between the actual and predicted price.

    Formula:
    score = max(0, 100 - (|actual - predicted| / actual * 100))

    Parameters
    ----------
    actual_price : float
        The actual sale price.
    predicted_price : float
        The model's predicted sale price.

    Returns
    -------
    float
        The calculated custom score (range: 0 to 100, inclusive).
    """
    if actual_price != 0:
        price_diff_percentage = abs(actual_price - predicted_price) / actual_price
        return max(0, 100 - (price_diff_percentage * 100))
    else:
        return 0

# Generate predictions
y_pred = model.predict(X_test)

# Compute custom scores
price_scores = []
for i in range(len(y_test)):
    actual_price = y_test.iloc[i]
    pred_price = max(0, y_pred[i])  # Predicted price cannot be negative in real terms

    price_score = calculate_price_score(actual_price, pred_price)
    price_scores.append(price_score)

    # Print predicted vs actual for the first 5
    if i < 5:
        print(f'Record {i+1}:')
        print(f'  Predicted Price: {pred_price:.2f}, Actual Price: {actual_price:.2f}')

# Calculate average price score
average_price_score = np.mean(price_scores)

print(f'Average Price Score: {average_price_score:.2f}')

