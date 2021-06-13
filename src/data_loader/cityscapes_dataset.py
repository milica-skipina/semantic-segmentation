import os
import random

from PIL import Image
import matplotlib.pyplot as plt
from cityscapesscripts.preparation.json2labelImg import createLabelImage
from cityscapesscripts.helpers.annotation import Annotation
from torch.utils.data import Dataset
from torchvision import transforms


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
        x = random.randint(0, image.size[0] - 256)
        y = random.randint(0, image.size[1] - 256)

        image = image.crop((x, y, x + 256, y + 256))
        label = label.crop((x, y, x + 256, y + 256))

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(label)
        # plt.show()

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

        return images_paths[:200], labels_paths[:200]
