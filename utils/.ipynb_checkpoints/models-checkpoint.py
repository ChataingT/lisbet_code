
import torch
import torch.nn as nn
from torch.utils.data import  Dataset

# Sample dataset implementation
class AutismDataset(Dataset):
    def __init__(self, X, y,idx_video=None, device='cpu'):
        """
        X: np.array of shape (num_samples, num_features)
        y: np.array of shape (num_samples,)
        """
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        self.idx_video = idx_video

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.idx_video[idx]
    
# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
class OptimizedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(OptimizedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    


class CNN(nn.Module):
    def __init__(self, input_size, sequence_length, num_filters, kernel_size, pool_size, hidden_size, output_size):
        super(CNN, self).__init__()
        
        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(
            in_channels=input_size,  # Number of keypoints per frame (34)
            out_channels=num_filters,  # Number of filters
            kernel_size=kernel_size  # Width of each filter
        )
        self.relu = nn.ReLU()
        
        # MaxPooling Layer
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        
        # Calculate the reduced sequence length after convolution and pooling
        conv_output_length = sequence_length - kernel_size + 1  # After convolution
        pooled_length = conv_output_length // pool_size         # After pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * pooled_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)  # Rearrange to (batch_size, input_size, sequence_length)
        x = self.conv1(x)  # Convolution
        x = self.relu(x)   # Activation
        x = self.pool(x)   # MaxPooling
        x = x.flatten(start_dim=1)  # Flatten for fully connected layers
        x = self.relu(self.fc1(x))  # Fully connected layer
        x = self.sigmoid(self.fc2(x))  # Output layer
        return x
    


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        """
        LSTM model for sequence classification.

        Args:
            input_size (int): Number of features per time step (e.g., 34 keypoints).
            hidden_size (int): Number of LSTM hidden units.
            output_size (int): Number of output units (e.g., 1 for binary classification).
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super(LSTMModel, self).__init__()
        
        # Define LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, input_size)
            dropout=dropout
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Output activation (use sigmoid for binary classification)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Pass through LSTM
        out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state for classification
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        
        # Apply sigmoid activation for binary classification
        out = self.sigmoid(out)
        return out
