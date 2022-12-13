#!/usr/bin/env python3
import pickle
from copy import deepcopy
from random import randint, uniform
from statistics import mean
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from model import Model
from dataset import Dataset


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # Create the dataset
    dataset = Dataset(3000, 0.9, 14, sigma=3)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    # for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):
    #     print('batch ' + str(batch_idx) + ' has xs of size ' + str(xs_ten.shape))
    # exit(0)

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model()
    model.to(device)  # Move model to cpu if exists

    learning_rate = 0.01
    maximum_num_epochs = 500
    termination_loss_threshold = 7.5
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    idx_epoch = 0
    # training the model

    while True:

        losses = []
        epoch_losses = []
        for batch_idx, (xs_ten, ys_ten_labels) in tqdm(enumerate(loader), total=len(loader), desc='Training batchesfor Epoch ' + str(idx_epoch)):

            xs_ten = xs_ten.to(device)
            ys_ten_labels = ys_ten_labels.to(device)

            # get output from the model, given the inputs
            ys_predicted = model.forward(xs_ten)

            # get loss for the predicted output
            loss = criterion(ys_predicted, ys_ten_labels)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            losses.append(loss.data.item())

        epoch_loss = mean(losses)
        epoch_losses.append(epoch_loss)

        idx_epoch += 1
        if idx_epoch > maximum_num_epochs:
            break
        elif epoch_loss < termination_loss_threshold:
            print('Reached target loss')
            break

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    ys_ten_predicted = model.forward(dataset.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    # plt.clf()
    plt.plot(dataset.xs_np, dataset.ys_np_labels, 'g.', label='labels')
    plt.plot(dataset.xs_np, ys_np_predicted, 'rx', label='Predictions')
    plt.legend(loc='best')

    plt.figure()
    plt.plot(0, len(epoch_losses), )

    plt.show()


if __name__ == "__main__":
    main()
