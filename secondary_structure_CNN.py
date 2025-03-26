# Install libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from torch.optim.lr_scheduler import LambdaLR
import datetime  # For timestamping
import os

### DEFINE FUNCTIONS BEGIN ###

# One hot encoding
def pad_sequence(sequence, max_length, padding_value=0):
"""
Pad a sequence with value 0 to ensure they all have the same length for inputs into neural network
    
    while len(sequence) < max_length:
        sequence.append(padding_value)
    return sequence
"""

def one_hot_encode_sequence(sequence):
    """
    Enumerates a sequence of proteins into a binary vector where each amino acid is represented by a binary vector.
    
    """
    
    return np.array([[1 if i == aa_to_index[aa] else 0 for i in range(len(amino_acids))] for aa in sequence])

def one_hot_encode_dssp3(dssp3_sequence):
    """
    Takes the input of DSSP3 labelled proteins and encodes them using the dssp3_to_index mapping dictionary.
    This results in secondary structure classes being enumerated to numerical labels (['H','E','C'] to [0,1,2]) 
    """
    return np.array([dssp3_to_index[label] for label in dssp3_sequence])

# Learning rate scheduler
def lr_lambda(epoch):
    """
    Adjusts learning rate.
    Uses learning_rate_schedule to store learning rates against epoch number as milestone:learning rate pairs, then changes learning rate to the appropriate key for the milestone
    if the epoch exceeds the given milestone in the dictionary.
    """
    for milestone in sorted(learning_rate_schedule.keys(), reverse=True):
        if epoch >= milestone:
            return learning_rate_schedule[milestone] / learning_rate_schedule[0]
    return 1.0

# Model architecture


class ResidueLevelCNNLSTM(nn.Module):
    
    """
    Defines model architecture of:
        
    1. Input layer
    2. 3x Convulutional layer with filter sizes of 3, 5, and 7   
    3. 1x Batch normalisation layer
    4. 1x Bidirectional LSTM layer
    5. 1x Dense layer (fully connected)
    6. 1x Output layer
    """
    def __init__(self, input_size, num_classes):
        super(ResidueLevelCNNLSTM, self).__init__()
        self.conv3 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_size, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_size, 64, kernel_size=7, padding=3)
        self.batch_norm = nn.BatchNorm1d(192)
        self.lstm = nn.LSTM(input_size=192, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        
        """
        Applies 1 dimensional convolutions using the three convolutional layers to extract local patterns
        ReLU activation function
        """
    
        x3 = F.relu(self.conv3(x))
        x5 = F.relu(self.conv5(x))
        x7 = F.relu(self.conv7(x))
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)  # Back to [batch_size, sequence_length, channels]
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x



"""DEFINE FUNCTIONS END"""

# define working directory, ensure data is in current working directory

cwd = os.getcwd()

### SCRIPT BEGIN ###

# Load data
file_path = os.path.join(cwd, 'cb513.csv')
data = pd.read_csv(file_path)
df = pd.DataFrame(data)

# One-hot encode amino acid sequences and DSSP3 protein structure classes
amino_acids = list("ARNDCQEGHILKMFPSTWYVXUZ")
dssp3_classes = list("HEC")
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
dssp3_to_index = {label: idx for idx, label in enumerate(dssp3_classes)}
df['one_hot_sequence'] = df['input'].apply(one_hot_encode_sequence)
df['one_hot_dssp3'] = df['dssp3'].apply(one_hot_encode_dssp3)


# Pad sequences to maintain dimensions during convolution
# find longest sequence 
max_length = max(df['one_hot_sequence'].apply(len))

# Pad sequences to max length sequence
df['one_hot_sequence'] = df['one_hot_sequence'].apply(lambda seq: pad_sequence(seq.tolist(), max_length, [0] * len(amino_acids)))
df['one_hot_dssp3'] = df['one_hot_dssp3'].apply(lambda seq: pad_sequence(seq.tolist(), max_length, -1))

# Define feature labels
X = np.stack(df['one_hot_sequence'].values)
y = np.stack(df['one_hot_dssp3'].values)

# Split datasets into train and test data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# In the CB513 dataset, class labels are unbalanced, so oversampling is performed to prevent class bias.
# Flatten labels for oversampling, oversampler expects a 2d array
y_train_flat = y_train.flatten()
X_train_flat = X_train.reshape(-1, X_train.shape[2])

# Oversample coils and sheets to be proportional with helices to prevent class imbalance. The maximum class label size (helices = 50374) is used to set other labels to.
# index -1 represents padding so this is ignored.
target_sample_size = 50374
sampling_strategy = {class_label: target_sample_size for class_label in Counter(y_train_flat).keys() if class_label != -1}
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_resampled_flat, y_train_resampled_flat = ros.fit_resample(X_train_flat, y_train_flat)

# Reshape features back to original dimensions expected by the input layer [batch_size, sequence_length, num_dimensions (23 amino acids)]
sequence_length = X_train.shape[1]
feature_dim = X_train.shape[2]
valid_batch_size = X_train_resampled_flat.shape[0] // sequence_length
target_rows = valid_batch_size * sequence_length
X_train_resampled_flat = X_train_resampled_flat[:target_rows]
y_train_resampled_flat = y_train_resampled_flat[:target_rows]
X_train_resampled = X_train_resampled_flat.reshape(valid_batch_size, sequence_length, feature_dim)
y_train_resampled = y_train_resampled_flat.reshape(valid_batch_size, sequence_length)

# Model, optimiser, and scheduler
input_size = X_train.shape[2] # set input size to the number of feature dimensions
num_classes = len(dssp3_classes) # set number of classes to number of classes in dssp3 (3)
model = ResidueLevelCNNLSTM(input_size, num_classes) # create instance of earlier defined model architecture
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]), ignore_index=-1) # Set loss to have no weight bias, and ignore padding
learning_rate_schedule = {0: 0.0025, 150: 0.0015, 170: 0.001, 180: 0.00050, 210: 0.00025, 230: 0.0001} # dictionary for learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_schedule[0]) # utilise Adam optomiser
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda) #  set learning rate scheduler (lm_lambda function)

# Initialise log storage
logs = []

# Training loop
epochs = 400

# Loads dataset into a PyTorch DataLoader
train_loader = DataLoader(TensorDataset(torch.tensor(X_train_resampled, dtype=torch.float32),
                                        torch.tensor(y_train_resampled, dtype=torch.long)),
                          batch_size=32, shuffle=True)

for epoch in range(epochs):
    model.train() # set training mode
    epoch_loss = 0 # initialise loss value

    # transpose [batch, channels, feature_dim] to [batch, channels, seq_len] to match expected CNN input
    # flatten outputs to 2D as nn.CrossEntropyLoss expects 2D predictions and 1D targets
    for inputs, labels in train_loader:
        inputs = inputs.permute(0, 2, 1)
        outputs = model(inputs)
        outputs = outputs.reshape(-1, num_classes)
        labels = labels.view(-1)

        # calculate loss, backpropagate, update model weights
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() # Update loss value with accumulated loss
    scheduler.step() # adjust learning rate based on step

    # Begin validation loop
    
    # Validation metrics
    val_preds, val_labels = [], []

    # Enter evaluation mode, no gradient tracking, no drop out
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                         torch.tensor(y_val, dtype=torch.long)),
                                           batch_size=32, shuffle=False):
            X_batch = X_batch.permute(0, 2, 1)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, axis=2).numpy()
            val_preds.extend(preds.flatten())
            val_labels.extend(y_batch.numpy().flatten())
    # mask labels where label is -1 to avoid padding                                           
    val_mask = np.array(val_labels) != -1
    val_accuracy = accuracy_score(np.array(val_labels)[val_mask], np.array(val_preds)[val_mask])

    logs.append({'epoch': epoch + 1, 'loss': epoch_loss, 'validation_accuracy': val_accuracy})
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test set evaluation
test_preds, test_labels = [], []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                                     torch.tensor(y_test, dtype=torch.long)),
                                       batch_size=32, shuffle=False):
        X_batch = X_batch.permute(0, 2, 1)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, axis=2).numpy()
        test_preds.extend(preds.flatten())
        test_labels.extend(y_batch.numpy().flatten())
test_mask = np.array(test_labels) != -1
test_accuracy = accuracy_score(np.array(test_labels)[test_mask], np.array(test_preds)[test_mask])

print(f"Test Set Accuracy: {test_accuracy:.4f}")

# Print the classification report
print("Classification Report on Test Set:")
print(classification_report(np.array(test_labels)[test_mask], np.array(test_preds)[test_mask], target_names=dssp3_classes, zero_division=0))

# Save logs to CSV
logs_df = pd.DataFrame(logs)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(cwd, f"training_logs_{timestamp}.csv")
logs_df.to_csv(output_path, index=False)
print(f"Training logs saved to {output_path}")

# Save model
torch.save(model.state_dict(), os.path.join(cwd, 'model_state.pth')

