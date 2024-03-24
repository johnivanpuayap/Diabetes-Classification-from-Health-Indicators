import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW

# Load the diabetes dataset
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

texts = df['Diabetes_012'].values
labels = df[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost','GenHlth', 'MentHlth','DiffWalk','Sex', 'Age', 'Education', 'Income']].values  # Assume you have 3 labels

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.3)

# Filter out any non-string or None values
train_texts = [str(text) for text in train_texts if text is not None]
test_texts = [str(text) for text in test_texts if text is not None]

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")

# Tokenize input
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert inputs to PyTorch tensors
train_inputs = torch.tensor(train_encodings['input_ids'])
train_masks = torch.tensor(train_encodings['attention_mask'])
train_labels = torch.tensor(train_labels.tolist())

test_inputs = torch.tensor(test_encodings['input_ids'])
test_masks = torch.tensor(test_encodings['attention_mask'])
test_labels = torch.tensor(test_labels.tolist())

# Load pre-trained model (weights)
model = AutoModelForSequenceClassification.from_pretrained('medicalai/ClinicalBERT', num_labels=train_labels.size(1))

# Set the batch size for training
batch_size = 16

# Create the DataLoader for training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for testing set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Set the learning rate and number of training epochs
learning_rate = 2e-5
num_epochs = 3

# Set the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        # Unpack the batch
        input_ids, attention_mask, labels = batch

        # Clear gradients
        model.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters
        optimizer.step()
        scheduler.step()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)

    # Evaluation loop
    model.eval()
    total_eval_loss = 0

    for batch in test_dataloader:
        # Unpack the batch
        input_ids, attention_mask, labels = batch

        # Disable gradient calculation
        with torch.no_grad():
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()

    # Calculate average evaluation loss for the epoch
    avg_eval_loss = total_eval_loss / len(test_dataloader)

    # Print the average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f} - Average Eval Loss: {avg_eval_loss:.4f}")
