#!/usr/bin/env python3

import random
from statistics import mean
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from Aula12.data_visualizer import DataVisualizer
from Aula12.model import Model
from dataset import Dataset
import glob


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = Model()
    model.to(device)  # Move model to cpu if exists

    learning_rate = 0.01
    maximum_num_epochs = 150
    termination_loss_threshold = 0.001
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    # Create the dataset
    dataset_path = '../datasets/catsxdogs/train'
    image_filenames = glob.glob(dataset_path + '/*.jpg')

    image_filenames = random.sample(image_filenames, k=900)

    train_image_filenames, test_image_filenames = train_test_split(image_filenames, test_size=0.2)

    dataset_train = Dataset(train_image_filenames)
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)

    tensor_to_pil_image = transforms.ToPILImage()
    for image_t, label_t in loader_train:
        print(image_t.shape)

        num_images = image_t.shape[0]
        image_idxs = random.sample(range(0, num_images), k=25)
        print(image_idxs)

        fig = plt.figure()
        for subplot_idx, image_idx in enumerate(image_idxs, start=1):

            image_pil = tensor_to_pil_image(image_t[image_idx, :, :, :])
            ax = fig.add_subplot(5, 5, subplot_idx)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            label = label_t[image_idx].data.item()
            class_name = 'dog' if label == 0 else 'cat'
            ax.set_xlabel(class_name)

            plt.imshow(image_pil)

        plt.show()
        exit(0)

    dataset_test = Dataset(500, 0.9, 14, sigma=3)
    # loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=256, shuffle=True)

    # for batch_idx, (xs_ten, ys_ten_labels) in enumerate(loader):
    #     print('batch ' + str(batch_idx) + ' has xs of size ' + str(xs_ten.shape))
    # exit(0)

    for image_t, label_t in loader_train:
        model.forward(image_t)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    loss_visualizer = DataVisualizer('Loss')
    loss_visualizer.draw([0, maximum_num_epochs], [termination_loss_threshold, termination_loss_threshold], layer='threshold', marker='--', markersize=1, color=[0.5, 0.5, 0.5], alpha=1, label='threshold', x_label='Epochs', y_label='Loss')

    model.to(device)
    idx_epoch = 0
    # training the model
    epoch_losses = []
    while True:

        losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train), desc='Training batchesfor Epoch ' + str(idx_epoch)):

            image_t = image_t.to(device)
            label_t = label_t.to(device)

            # get output from the model, given the inputs
            label_t_predicted = model.forward(image_t)

            # get loss for the predicted output
            loss = loss_function(label_t_predicted, label_t)

            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            optimizer.step()

            losses.append(loss.data.item())

        epoch_loss = mean(losses)
        epoch_losses.append(epoch_loss)

        loss_visualizer.draw([list(range(0, len()), [termination_loss_threshold, termination_loss_threshold],
                             layer='threshold', marker='--', markersize=1, color=[0.5, 0.5, 0.5], alpha=1,
                             label='threshold', x_label='Epochs', y_label='Loss')

        idx_epoch += 1
        if idx_epoch > maximum_num_epochs:
            break
        elif epoch_loss < termination_loss_threshold:
            print('Reached target loss')
            break

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------




if __name__ == "__main__":
    main()
