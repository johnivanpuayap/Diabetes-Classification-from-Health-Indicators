from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset dataset
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Split the dataset into features (X) and target variable (y)
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Resampling')
# Apply SMOTEENN to balance the training data
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
print('Finished resampling')

# Plotting pie charts side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot original class distribution
original_class_distribution = Counter(y_train)
axs[0].pie(original_class_distribution.values(), labels=original_class_distribution.keys(), autopct='%1.1f%%')
axs[0].set_title('Original Class Distribution')

# Plot class distribution after resampling
resampled_class_distribution = Counter(y_train_resampled)
axs[1].pie(resampled_class_distribution.values(), labels=resampled_class_distribution.keys(), autopct='%1.1f%%')
axs[1].set_title('Class Distribution after SMOTEENN')

plt.show()

print('Training the Random Forest Classifier')
# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

print('Predicting')
# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)