#!/usr/bin/env python3
import pickle
from copy import deepcopy
from random import randint, uniform
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
    dataset = Dataset(3000, 0.3, 14)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=True)

    # for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):
    #     print('batch ' + str(batch_idx) + ' has xs of size ' + str(xs_ten.shape))
    # exit(0)

    # Define hyper parameters
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model()
    model.to(device)  # Move model to cpu if exists

    learning_rate = 0.01
    maximum_num_epochs = 50
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    idx_epoch = 0
    # training the model

    while True:

        for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):

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

        idx_epoch += 1

        if idx_epoch > maximum_num_epochs:
            break

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    ys_ten_predicted = model.forward(dataset.xs_ten.to(device))
    ys_np_predicted = ys_ten_predicted.cpu().detach().numpy()

    # plt.clf()
    plt.plot(dataset.xs_np, dataset.ys_np_labels, 'go', label='labels')
    plt.plot(dataset.xs_np, ys_np_predicted, '--r', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
