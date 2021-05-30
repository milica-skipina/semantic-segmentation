import os
from PIL import Image
import matplotlib.pyplot as plt
from cityscapesscripts.preparation.json2labelImg import createLabelImage
from cityscapesscripts.helpers.annotation import Annotation


class Dataset:
    def __init__(self, data_path=None, labeled_data_path=None):
        self.data_path = data_path
        self.labeled_data_path = labeled_data_path

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

        x = []
        y = []

        for image_path, label_path in zip(images_paths, labels_paths):
            image = Image.open(image_path)
            x.append(image)
            # image = image.convert('RGB')
            # plt.imshow(image)
            # plt.show()

            annotation = Annotation()
            annotation.fromJsonFile(label_path)
            labeled_image = createLabelImage(annotation, 'categoryId')
            y.append(labeled_image)
            # plt.imshow(labeled_image)
            # plt.show()

        return x, y
