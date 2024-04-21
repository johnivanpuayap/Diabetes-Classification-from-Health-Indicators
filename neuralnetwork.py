import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from keras.callbacks import EarlyStopping


# Load your dataset
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Apply SMOTE to balance the training data")
# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Plotting pie charts side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot original class distribution
original_class_distribution = Counter(y_train)
axs[0].pie(original_class_distribution.values(), labels=original_class_distribution.keys(), autopct='%1.1f%%')
axs[0].set_title('Original Class Distribution')

# Plot class distribution after resampling
resampled_class_distribution = Counter(y_train_smote)
axs[1].pie(resampled_class_distribution.values(), labels=resampled_class_distribution.keys(), autopct='%1.1f%%')
axs[1].set_title('Class Distribution after SMOTE')


# Define a function to create the model
def create_model(layers=1, activation='relu', neurons=64, batch_size=32, epochs=10):
    model = Sequential()
    model.add(Input(shape=(X_train_smote.shape[1],)))
    model.add(Dense(units=neurons, activation=activation))
    for _ in range(layers):
        model.add(Dense(units=neurons, activation=activation))
    model.add(Dense(3, activation='softmax'))  # 3 output classes (0, 1, 2) for no diabetes, prediabetes, and diabetes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    history = model.fit(X_train_smote, y_train_smote, batch_size=batch_size, epochs=epochs, verbose=0, validation_split=0.2, callbacks=[early_stopping])
    return model, history

print("Create a KerasClassifier based on the function")

# Create a KerasClassifier based on the function
# model is similar to the build_fn parameter in keras.wrappers.scikit_learn KerasClassifier
model = KerasClassifier(model=create_model, verbose=0)

# Define the grid of hyperparameters to search
param_grid = {
    'model__activation': ['relu', 'tanh', 'sigmoid'],
    'model__neurons': [32, 64, 128],
    'model__layers': [1, 2, 3],
}

print("Perform GridSearchCV")
# Perform GridSearchCV
# cv: This parameter determines the cross-validation splitting strategy
# n_jobs: Number of jobs to run in parallel, we set it to -1 to use all processors
# scoring: The scoring method used to evaluate the model
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, scoring='accuracy', verbose=2)

# grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=5, cv=3, verbose=0)
grid_search.fit(X_train_smote, y_train_smote)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Access the history object from the best_estimator_ attribute of GridSearchCV
best_model_history = grid_search.best_estimator_.model.history

# Retrieve the number of epochs
num_epochs = len(best_model_history.history['loss'])
print("Number of epochs:", num_epochs)

# Predictions
y_pred = grid_search.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))