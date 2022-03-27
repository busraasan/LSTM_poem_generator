import torch
import torch.nn as nn
import os
from collections import Counter
import numpy as np
from data_handler import DataHandler
from model import LSTM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--batch', type=int, default=16, help='Batch Size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
opt = parser.parse_args()

batch_size = opt.batch
epochs = opt.epoch
lr = opt.lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Parameters for NN
embedding_vector_dim = 64 #seq_size

handler = DataHandler('shakespeare.txt', batch_size=batch_size, seq_size=embedding_vector_dim)
model = LSTM(num_of_samples=len(handler.embeddings), lstm_size=64)
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def train(model):
    iteration = 0
    for epoch in range(epochs):

        batches = handler.batcher() #creating batches
        
        h, c = torch.zeros(1, batch_size, model.lstm_size), torch.zeros(1, batch_size, model.lstm_size) #init h and c states
        h = h.to(device)
        c = c.to(device)
        
        for x, y in batches:
            iteration += 1
            model.zero_grad() #clear gradients
            x_ = torch.tensor(x).to(device)
            y_ = torch.tensor(y).to(device)
            predictions, (h, c) = model(x_, (h,c))
            h = h.detach()
            c = c.detach()
            loss = loss_function(predictions.transpose(1,2).to(device), y_)
            #total_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        
            if(iteration%100 == 0):
                print(f'Loss: [{loss.item()}]')

        print(f'Epoch [{epoch + 1}/{epochs}]')

def predict_text(model, words, top_k=5):
    model.eval()

    h, c = torch.zeros(1, 1, model.lstm_size), torch.zeros(1, 1, model.lstm_size)
    h = h.to(device)
    c = c.to(device)

    words = handler.word_parser(words)

    for w in words:
        x_i = torch.tensor([[handler.word_embedder(w)]]).to(device)
        output, (h, c) = model(x_i, (h, c))
    
    # _, top_x_i = torch.topk(output[0], k=1)
    # choices = top_x_i.tolist()
    # choice = choices[0]

    top_x_i = torch.argmax(output[0])
    top_x_i = top_x_i.item()

    words.append(handler.word_decoder(top_x_i))
    
    for _ in range(100):
        x_i = torch.tensor([[top_x_i]]).to(device)
        output, (h, c) = model(x_i, (h, c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        top_x_i = np.random.choice(choices[0])
        words.append(handler.word_decoder(top_x_i))

    print(' '.join(words))


if __name__ == "__main__":
    train(model)
    predict_text(model, "It shall be resolved")