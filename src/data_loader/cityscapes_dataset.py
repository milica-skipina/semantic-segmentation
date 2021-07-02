import os
import random

import numpy as np
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.preparation.json2labelImg import createLabelImage

TRAIN_H = 256
TRAIN_W = 256


def load_data(data_path, labeled_data_path):
    images_paths = []
    labels_paths = []
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(subdir, file)
                images_paths.append(image_path)

    for subdir, dirs, files in os.walk(labeled_data_path):
        for file in files:
            if file.endswith('.json'):
                json_file_path = os.path.join(subdir, file)
                labels_paths.append(json_file_path)

    images_paths = sorted(images_paths)
    labels_paths = sorted(labels_paths)

    return images_paths, labels_paths


class CityscapesDataset(Dataset):
    def __init__(self, data_path: str, labeled_data_path: str, dataset_len: int, transform: bool, is_test=False):
        self.data_path = data_path
        self.labeled_data_path = labeled_data_path
        self.images_paths, self.labels_paths = load_data(data_path, labeled_data_path)
        self.transform = transform
        self.data_len = len(self.images_paths)
        self.len = dataset_len
        self.is_test = is_test

    @staticmethod
    def transform_image(image, ground_truth):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        ground_truth = torch.from_numpy(np.array(ground_truth, dtype=np.uint8)).long()
        return transform(image), ground_truth

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_idx = idx % self.data_len
        # print('index {}, data_idx {}'.format(idx, data_idx))
        image_path = self.images_paths[data_idx]
        image = Image.open(image_path)
        label_path = self.labels_paths[data_idx]
        annotation = Annotation()
        annotation.fromJsonFile(label_path)
        label = createLabelImage(annotation, 'categoryId')
        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()
        if self.is_test:
            images = []
            labels = []
            for x in range(2048 // TRAIN_W):
                for y in range(1024 // TRAIN_H):
                    im = image.crop((x * TRAIN_W, y * TRAIN_H, (x + 1) * TRAIN_W, (y + 1) * TRAIN_H))
                    lb = label.crop((x * TRAIN_W, y * TRAIN_H, (x + 1) * TRAIN_W, (y + 1) * TRAIN_H))
                    # plt.imshow(im)
                    # plt.show()
                    # plt.imshow(lb)
                    # plt.show()
                    if self.transform:
                        im, lb = self.transform_image(im, lb)
                    images.append(im)
                    labels.append(lb)
            return images, labels
        else:
            # random crop
            x = random.randint(0, image.size[0] - TRAIN_W)
            y = random.randint(0, image.size[1] - TRAIN_H)

            image = image.crop((x, y, x + 256, y + TRAIN_W))
            label = label.crop((x, y, x + 256, y + TRAIN_H))

            # plt.imshow(image)
            # plt.show()
            # plt.imshow(label)
            # plt.show()

            if self.transform:
                image, label = self.transform_image(image, label)

            return image, label


if __name__ == '__main__':
    loader = torch.utils.data.DataLoader(CityscapesDataset('../../data/raw/leftImg8bit/train',
                                                           '../../data/raw/gtFine/train',
                                                           dataset_len=20, transform=True), batch_size=2)

    for batch_id, (img, gt) in enumerate(loader):
        print(img.shape)
        print(gt.shape)
        print('\n')

    print(len(loader))
