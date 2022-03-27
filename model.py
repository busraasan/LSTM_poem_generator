import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_of_samples, seq_size=32, lstm_size=64):
        super(LSTM, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(num_of_samples, seq_size) #A simple lookup table that stores embeddings of a fixed dictionary and size.
                                                                    #Stores word embeddings and retrieve them using indices.
        self.lstm = nn.LSTM(seq_size, lstm_size, batch_first=True)
        self.linear1 = nn.Linear(lstm_size, 128)
        self.dropout1 = nn.Dropout(0.4)
        self.linear2 = nn.Linear(128,128)
        self.linear3 = nn.Linear(128, lstm_size)
        self.dense = nn.Linear(lstm_size, num_of_samples)

    def forward(self, x, prev_state):
        embeddings = self.embedding(x)
        output, current_state = self.lstm(embeddings, prev_state)
        #output = self.linear1(output)
        #output = F.relu(output)
        output = self.dropout1(output)
        #output = self.linear3(output)
        predictions = self.dense(output)
        return predictions, current_state
 