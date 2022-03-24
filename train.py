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
embedding_vector_dim = 32

handler = DataHandler('shakespeare.txt', batch_size=batch_size)
model = LSTM(num_of_samples=len(handler.embeddings), feature_size=embedding_vector_dim, lstm_size=32)
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
        

def predict_text(model, input_string, next_words=100):

    model.eval()
    h, c = torch.zeros(1, 1, model.lstm_size), torch.zeros(1, 1, model.lstm_size)
    h = h.to(device)
    c = c.to(device)

    parsed_input = handler.word_parser(input_string)
    input_embeddings = torch.tensor([[[handler.word_embedder(w) for w in parsed_input]]])
    generated_text = []
    for i in range(len(input_embeddings[0][0].tolist())):
        generated_text.append(input_embeddings[0][0].tolist()[i])

    for embedding in input_embeddings:
        x_i = embedding.clone().detach()
        x_i = x_i.to(device)
        output, (h,c) = model(x_i, (h,c))


    print(output[0][-1].cpu().detach().numpy())
    pred = np.argmax(output[0][-1].cpu().detach().numpy())
    print("pred indices: ", pred)
    generated_text.append(pred)
    current_pred = pred
    
    for _ in range(next_words):
        print("predicting..")
        x_i = torch.tensor([[current_pred]]).to(device)
        output, (h,c) = model(x_i, (h,c))
        current_pred = np.argmax(output[0][-1].cpu().detach().numpy())
        generated_text.append(current_pred)

    print("generateeee<3")
    print(generated_text)
    decoded_text = [handler.word_decoder(w) for w in generated_text]
    print(' '.join(decoded_text))

def generate_text(model, words, top_k=5):
    model.eval()

    state_h, state_c = torch.zeros(1, 1, model.lstm_size), torch.zeros(1, 1, model.lstm_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    words = handler.word_parser(words)

    for w in words:
        ix = torch.tensor([[handler.word_embedder(w)]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(handler.word_decoder(choice))
    
    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = model(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(handler.word_decoder(choice))

    print(' '.join(words))


if __name__ == "__main__":
    train(model)
    generate_text(model, "You are all resolved rather to die than to famish?")