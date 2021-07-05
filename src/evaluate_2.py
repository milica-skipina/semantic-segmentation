from model.resnet import ResNet50
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import make_grid
import numpy as np
from torch import nn
from skimage import color
from matplotlib import pyplot as plt

from src.model.convAutoencoder import ConvAutoencoder
from src.model.inception import InceptionV3
from src.model.semanticSegmentation import SemanticSegmentationModel

DEVICE = torch.device('cpu')


def load_model(out_classes, path, model_type='resnet'):
    if model_type == 'resnet':
        model = ResNet50(out_classes=out_classes, is_autoencoder=False).to(DEVICE)
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
    elif model_type == 'inception':
        model = InceptionV3(out_classes=out_classes)
        model = nn.DataParallel(model)
        model.to(DEVICE)
        checkpoint_path = "/home/milica/Desktop/NN/reports/03_07_19_07/models/decoderCheckpoint.pth"

        try:
            loaded_checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(loaded_checkpoint['model_state'])
        except:
            print('error loading inception')
    elif model_type == 'autoencoder':
        loaded_model = torch.load(path)
        encoder = ConvAutoencoder()
        encoder = nn.DataParallel(encoder)
        model = SemanticSegmentationModel(encoder.module.encoder)
        model = nn.DataParallel(model)
        model.load_state_dict(loaded_model['model_state'])
        print('Model loaded.')
        print('autoencoder')
    else:
        return 'Unsupported model type'

    return model


def load_image(image_path):
    image = Image.open(image_path)
    return image


def PIL_to_tensor(PIL):
    transform = transforms.ToTensor()
    return transform(PIL)


def save_image(np_image, image_path):

    PIL_image = Image.fromarray((np_image * 255).astype(np.uint8))
    PIL_image.save(image_path)

    return PIL_image


def predict_from_image(model, image_path):
    pred_images = []

    with torch.no_grad():
        image = load_image(image_path)
        x = 0
        y = 0
        width = 256
        height = 256

        # 32 256x256 patches
        for patch_id in range(1, 33):
            patch = image.crop((x, y, x + width, y + height))
            #plt.imshow(patch)
            #plt.show()

            tensor_patch = PIL_to_tensor(patch)

            output = model(tensor_patch.unsqueeze(0))
            output = F.log_softmax(output, dim=1)
            output = torch.argmax(output, axis=1)

            #plt.imshow(output.data.cpu().numpy().transpose(1, 2, 0))
            #plt.show()

            pred_images.append(output)

            if patch_id % 8 == 0:
                x = 0
                y = y + height
            else:
                x = x + width

        prediction_grid = torch.squeeze(make_grid(pred_images, nrow=8, padding=0)[0, :]).data.cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image)
        ax2.imshow(prediction_grid)
        plt.show()

        # PIL_image = save_image(prediction_grid, f'predictions/{image_path}.png')

        #result_image = color.label2rgb(PIL_image, image)
        #plt.imshow(result_image)
        #plt.show()


if __name__ == '__main__':
    MODEL_PATH = '/home/milica/Desktop/NN/reports/03_07_19_07/models/decoderCheckpoint.pth'
    MODEL_PATH = '../models/1_segnet.pth'
    MODEL_PATH = '/home/milica/Desktop/NN/reports/03_07_10_20/decoderCheckpoint.pth'

    model = load_model(out_classes=8, path=MODEL_PATH, model_type='autoencoder')
    model.eval()

    predict_from_image(model, '../data/raw/leftImg8bit/val/frankfurt/frankfurt_000001_007285_leftImg8bit.png')
    predict_from_image(model, '../data/raw/leftImg8bit/val/frankfurt/frankfurt_000001_010156_leftImg8bit.png')
    predict_from_image(model, '../data/raw/leftImg8bit/val/frankfurt/frankfurt_000001_070099_leftImg8bit.png')
    predict_from_image(model, '../data/raw/leftImg8bit/val/frankfurt/frankfurt_000000_000576_leftImg8bit.png')


