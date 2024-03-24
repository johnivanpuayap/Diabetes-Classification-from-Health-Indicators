import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras_nlp import BERTClassifier
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

# Concatenate multiple features into a single text string
df['concatenated_features'] = df['HighBP'] + ' ' + df['HighChol'] + ' ' + df['CholCheck']  + ' ' + df['BMI']  + ' ' + df['Smoker']  + ' ' + df['Stroke']  + ' ' + df['HeartDiseaseorAttack'] + ' ' + df['PhysActivity']  + ' ' + df['Fruits']  + ' ' + df['Veggies']  + ' ' + df['HvyAlcoholConsump']  + ' ' + df['AnyHealthCare'] + df['NoDocbcCost']  + ' ' + df['GenHlth'] + df['MentHlth']  + ' ' + df['DiffWalk']  + ' ' + df['Sex']  + ' ' + df['Age'] +   + ' ' + df['Education'] +   + ' ' + df['Income']

# Prepare features (concatenated text) and labels
features = df['concatenated_features'].tolist()
labels = df['Diabetes_012'].tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

print('Applying SMOTE')
# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize BERT Classifier
classifier = BERTClassifier.from_preset("bert_base_en", num_classes=3)  # Assuming 3 classes (0, 1, and 2)

print('Training the data')
# Fine-tune BERT on the training data
classifier.fit(x=X_train_smote, y=y_train_smote, batch_size=32)

# Compile the model with appropriate loss and optimizer
classifier.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=5e-5),
)

# Evaluate the model on the test set
loss, accuracy = classifier.evaluate(x=X_test, y=y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Make predictions
predictions = classifier.predict(x=X_test)

# Generate a classification report
class_names = ['Class 0', 'Class 1', 'Class 2']  # Adjust class names accordingly
report = classification_report(y_test, predictions.argmax(axis=1), target_names=class_names)
print("Classification Report:\n", report)