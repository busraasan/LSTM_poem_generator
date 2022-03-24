import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_of_samples, feature_size=64, seq_size=32, lstm_size=64):
        super(LSTM, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(num_of_samples, feature_size) #A simple lookup table that stores embeddings of a fixed dictionary and size.
                                                                    #Stores word embeddings and retrieve them using indices.
        self.lstm = nn.LSTM(feature_size, lstm_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)
        self.linear = nn.Linear(lstm_size, num_of_samples)

    def forward(self, x, prev_state):
        embeddings = self.embedding(x)
        output, current_state = self.lstm(embeddings, prev_state)
        output = self.dropout1(output)
        predictions = self.linear(output)
        return predictions, current_state
