import torch
from PIL import Image
from torchvision.transforms import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image_filenames):

        super().__init__()

        self.image_filenames = image_filenames
        self.num_images = len(self.image_filenames)

        self.labels = []
        for image_filename in self.image_filenames:
            self.labels.append(self.getClassFromFilename(image_filename))

        # Create a set of transformations
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    def __getitem__(self, index):   # Return specific element x, y given the index, of the dataset

        # Get the image
        image_pil = Image.open(self.image_filenames[index])

        image_t = self.transforms(image_pil)

        return image_t, self.labels[index]

    def __len__(self):  # Return the length of the dataset
        return self.num_images

    def getClassFromFilename(self, filename):

        parts = filename.split('/')
        part = parts[-1]

        parts = part.split('.')

        class_name = parts[0]
        # print('filename ' + filename + ' is a ' + class_name)

        if class_name == 'dog':
            label = 0
        elif class_name == 'cat':
            label = 1
        else:
            raise ValueError('Unknown class')

        return label
