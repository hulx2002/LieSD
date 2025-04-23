import torch
import torch.nn as nn
import numpy as np
from adan import Adan
import os

save_model_path = './saved_models'

def train_net(net, train_dataloader, num_epochs, device, train_print_every, dataset):
    print("Start training neural network.")
    if dataset == 'top_quark_tagging':
        criterion = nn.BCELoss(reduction="mean")
    elif dataset == 'mnist':
        criterion = nn.CrossEntropyLoss(reduction="mean")
    else:
        criterion = nn.MSELoss(reduction="mean")
    optimizer = Adan(net.parameters())
    for epoch in range(num_epochs):
        loss_list = []
        net.train()
        for x_raw, y_raw in train_dataloader:
            x_raw = x_raw.to(device)
            y_raw = y_raw.to(device)
            optimizer.zero_grad()
            if dataset != 'mnist':
                x = x_raw.reshape(x_raw.shape[0], -1)
                y = y_raw.reshape(y_raw.shape[0], -1)
            else:
                x = x_raw
                y = y_raw
            f = net(x)
            loss = criterion(f, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        if (epoch + 1) % train_print_every == 0:
            print(f'Epoch {(epoch + 1)}/{num_epochs}: loss={np.mean(loss_list)}')
    print("End training neural network.")
