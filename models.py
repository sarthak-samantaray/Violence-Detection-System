import cv2
import mediapipe as mp
import numpy as np
import torch
import threading
import torch.nn as nn
from ultralytics import YOLO
# Define LSTMModel class (no changes here)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_classes=3):
        super(LSTMModel, self).__init__()
        
        # Adjust input_size based on your actual input feature dimension
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        
        # Adjust input size for subsequent LSTM layers
        lstm_input_size = hidden_size * 2  # Because of bidirectional
        self.lstm2 = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        
        self.lstm3 = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(0.2)
        
        self.lstm4 = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(0.2)
        
        # Final fully connected layer: Adjust input size based on bidirectional LSTM output
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # hidden_size * 2 for bidirectional
        
    def forward(self, x):
        # LSTM layers with proper handling of bidirectional outputs
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = self.dropout4(x)
        
        # Take the last time step or average
        x = x[:, -1, :]  # Take last time step
        x = self.fc(x)  # Match the size of the last LSTM output
        
        return x
    

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=2):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.3)
        
        lstm_input_size = hidden_size * 2
        self.lstm2 = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout2 = nn.Dropout(0.3)
        
        # Reduced to 2 LSTM layers for better efficiency
        
        # Added attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context
        
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(self.batch_norm1(x.permute(0,2,1)).permute(0,2,1))
        
        x, _ = self.lstm2(x)
        x = self.dropout2(self.batch_norm2(x.permute(0,2,1)).permute(0,2,1))
        
        # Apply attention
        x = self.attention_net(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x