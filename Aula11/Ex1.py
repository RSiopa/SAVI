#!/usr/bin/env python3
import pickle
from copy import deepcopy
from random import randint, uniform
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


# model architecture
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    file = open('pts.pkl', 'rb')
    pts = pickle.load(file)
    file.close()

    x_train = np.array(pts['xs'], dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_train = np.array(pts['ys'], dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    # instantiate the model
    inputDim = 1  # takes variable 'x'
    outputDim = 1  # takes variable 'y'
    learningRate = 0.01
    epochs = 100

    model = linearRegression(inputDim, outputDim)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()

    # initialize loss and optimization
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------
    print(np.array(pts['xs']))
    print(x_train)
    # training the model
    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda())
            labels = Variable(torch.from_numpy(y_train).cuda())
        else:
            inputs = Variable(torch.from_numpy(x_train))
            labels = Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------

    with torch.no_grad():  # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
        else:
            predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        print(predicted)

    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
