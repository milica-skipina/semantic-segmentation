import os
import random

from PIL import Image
import matplotlib.pyplot as plt
from cityscapesscripts.preparation.json2labelImg import createLabelImage
from cityscapesscripts.helpers.annotation import Annotation
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def label2cat(label):
    # label[label == 3] = 1
    # label[label == 6] = 2
    # label[label == 7] = 3
    s = 0
    ret = Image.new(label.mode, label.size)
    for i in range(label.size[0]):
        for j in range(label.size[1]):
            if label.getpixel((i, j)) == 0:
                s += 1
                ret.putpixel((i, j), 0)
            elif label.getpixel((i, j)) == 3:
                ret.putpixel((i, j), 1)
            elif label.getpixel((i, j)) == 6:
                ret.putpixel((i, j), 2)
            elif label.getpixel((i, j)) == 7:
                ret.putpixel((i, j), 3)
    return ret, s


class CityscapesDataset(Dataset):
    def __init__(self, data_path=None, labeled_data_path=None, transform=None, target_transform=None):
        self.data_path = data_path
        self.labeled_data_path = labeled_data_path
        self.images_paths, self.labels_paths = self.load_data()

    def __len__(self):
        return len(self.images_paths)

    def transform(self, image):
        transform = transforms.Compose([
            transforms.ToTensor()]
        )
        return transform(image)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path)
        label_path = self.labels_paths[idx]
        annotation = Annotation()
        annotation.fromJsonFile(label_path)
        label = createLabelImage(annotation, 'categoryId')
        # random crop

        done = False
        it = 0
        while done == False and it < 1000:
            it += 1
            x = random.randint(0, image.size[0] - 256)
            y = random.randint(0, image.size[1] - 256)

            # print(x)
            # print(y)
            temp = image.crop((x, y, x + 256, y + 256))
            temp_label = label.crop((x, y, x + 256, y + 256))
            # plt.imshow(temp_label)
            # plt.show()
            temp_label, zero_sum = label2cat(temp_label)
            # plt.imshow(temp_label)
            # plt.show()
            # print(zero_sum)
            if zero_sum < 4*256*256/5:
                done = True
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()

        image = temp
        label = temp_label
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

    def load_data(self):
        images_paths = []
        labels_paths = []
        for subdir, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.png'):
                    image_path = os.path.join(subdir, file)
                    images_paths.append(image_path)

        for subdir, dirs, files in os.walk(self.labeled_data_path):
            for file in files:
                if file.endswith('.json'):
                    json_file_path = os.path.join(subdir, file)
                    labels_paths.append(json_file_path)

        images_paths = sorted(images_paths)
        labels_paths = sorted(labels_paths)

        return images_paths[:2], labels_paths[:2]
