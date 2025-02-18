import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Reads the CSV file into a Pandas DataFrame
data = pd.read_csv('test.csv')

# Select Relevant Columns: Keeps only the features required for the task
data = data[['Gender', 'Age', 'Type of Travel', 'Class','Inflight wifi service','Customer Type',
             'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'satisfaction']]

# Handle Missing Values: Removes rows with any NaN values
data = data.dropna()

# Label Encoding: Maps the satisfaction column to binary values (1 for satisfied, 0 for neutral/dissatisfied).
data['satisfaction'] = data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
#,'Inflight wifi service','Food and drink'

#Normalize Numerical Features:
    # Defines the numerical columns to scale.
    #Applies standardization to make the features have a mean of 0 and a standard deviation of 1.
numerical_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes','Inflight wifi service']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Converts categorical variables into binary (dummy) variables.
# Uses drop_first=True to avoid multicollinearity
data = pd.get_dummies(data, columns=['Gender', 'Type of Travel', 'Class', 'Customer Type'], drop_first=True)

# Shuffle Data: Randomizes the row order to improve model generalization.
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# X: All columns except the satisfaction target.
# y: The target column (satisfaction).
X = data.drop(columns='satisfaction')
y = data['satisfaction']

# Split Data into Training and Testing Sets:
#     70% of the data for training, 30% for testing.
#     random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert Data to NumPy Arrays: Required for TensorFlow compatibility.
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Build Neural Network:
#     First layer: 64 neurons with ReLU activation.
#     Second layer: 32 neurons with ReLU activation.
#     Output layer: 1 neuron with sigmoid activation for binary classification.
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the Model:
#     Optimizer: adam (efficient for most tasks).
#     Loss function: mse (preferably replaced with binary_crossentropy for binary classification).
#     Metrics: Tracks accuracy.
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# Early Stopping: Stops training if the validation loss does not improve for 5 epochs.
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the Model:
#     80% of the training data is used for training, 20% for validation.
#     Trains for up to 50 epochs or until early stopping triggers.
#     Batch size: Processes 32 samples at a time.
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the Model: Computes the loss and accuracy on the test set.
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Predict Probabilities: Outputs probabilities for each test sample.
y_pred_prob = model.predict(X_test)

# Calculate Optimal Threshold:
#     Computes precision, recall, and thresholds.
#     Selects the threshold that maximizes the product of precision and recall.
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
optimal_threshold = thresholds[np.argmax(precision * recall)]
print("Optimal Threshold:", optimal_threshold)

# Convert Probabilities to Binary Predictions: Applies the optimal threshold to generate binary predictions.
y_pred = (y_pred_prob > optimal_threshold).astype("int32")

# Confusion Matrix: Summarizes the true and false predictions for each class.
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Visualize Confusion Matrix: Creates a heatmap for easy interpretation of results.
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Neutral/Dissatisfied', 'Satisfied'], yticklabels=['Neutral/Dissatisfied', 'Satisfied'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report: Prints metrics like precision, recall, and F1 score for both classes.
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)


