import torch


# Model definition
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Define the structure of the neural network
        size_in, size_l1, size_l2, size_l3, size_out = 1, 64, 64, 64, 1
        # self.layer1 = torch.nn.Linear(1, 1)
        self.layer1 = torch.nn.Linear(size_in, size_l1)
        self.layer2 = torch.nn.Linear(size_l1, size_l2)
        self.layer3 = torch.nn.Linear(size_l2, size_l3)
        self.layer4 = torch.nn.Linear(size_l3, size_out)

    def forward(self, xs):

        # ys = self.layer1(xs)
        xs = torch.relu(self.layer1(xs))
        xs = torch.relu(self.layer2(xs))
        xs = torch.relu(self.layer3(xs))
        ys = self.layer4(xs)

        return ys
