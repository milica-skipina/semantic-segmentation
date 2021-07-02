import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from data_loader.cityscapes_dataset import CityscapesDataset, load_data
from model.resnet import ResNet50
from src.model.convAutoencoder import ConvAutoencoder
from src.model.semanticSegmentation import SemanticSegmentationModel

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = '../'
TRAIN_H = 256
TRAIN_W = 256


def prepare_data():
    test_dataset = CityscapesDataset('../data/raw/leftImg8bit/val', '../data/raw/gtFine/val', dataset_len=500,
                                     transform=True, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True)

    return test_loader


def to_img(y):
    y = y.view(3, 256, 256)
    return y


def load_model(model):
    if model == 'resnet':
        model_path = '../models/resnet.pth'
        loaded_model = torch.load(model_path)
        net = ResNet50(out_classes=8, is_autoencoder=False)
        net = nn.DataParallel(net)
        net.load_state_dict(loaded_model)
        print('Model loaded.')
    elif model == 'inception':
        model_path = '../models/inception.pth'
        loaded_model = torch.load(model_path)
        net = None
        net = nn.DataParallel(net)
        net.load_state_dict(loaded_model)
        print('Model loaded.')
    else:
        # model_path = '../models/autoencoder.pth'
        model_path = '/home/milica/Desktop/NN/reports/01_07_16_29/decoderCheckpoint.pth'
        loaded_model = torch.load(model_path)
        encoder = ConvAutoencoder()
        encoder = nn.DataParallel(encoder)
        net = SemanticSegmentationModel(encoder.module.encoder)
        net = nn.DataParallel(net)
        net.load_state_dict(loaded_model['model_state'])
        print('Model loaded.')

    print('Model loaded.')
    print('Number of model parameters:\t{}'.format(sum([p.data.nelement() for p in net.parameters()])))

    return net


class VAELoss:
    def __init__(self):
        pass

    def __call__(self, img, output, mu, logvar):
        """Adds the reconstruction loss and KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        mse_loss = F.mse_loss(img, output)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse_loss + kld


def run(args):
    model = load_model(args.model)
    test_loader = prepare_data()
    images_paths, labels_paths = load_data('../data/raw/leftImg8bit/val', '../data/raw/gtFine/val')

    start = time.time()

    save_dir = ROOT + 'results/'
    # for creating test dataset
    # save_dir = ROOT + 'data/processed/gtFine/val'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print('Test loop started.')
    model.eval()

    test_loss = 0

    for idx, (image, gt) in enumerate(test_loader):
        print(idx)
        x = 0
        y = 0
        result = Image.new("L", (2048, 1024))
        for img, lbl in zip(image, gt):
            img = img.to(DEVICE)
            lbl = lbl.to(DEVICE)

            output = model(img)
            loss = criterion(output, lbl.squeeze(dim=1))
            test_loss += loss.item()

            l = output.data.cpu().numpy().transpose(1, 2, 0).reshape(TRAIN_W, TRAIN_H)
            # for creating test dataset
            # l = lbl.data.cpu().numpy().transpose(1, 2, 0).reshape(TRAIN_W, TRAIN_H)
            l = Image.fromarray(l.astype('uint8'), 'L')

            result.paste(l, (x, y))
            y += TRAIN_H
            if y >= 1024:
                y = 0
                x += TRAIN_W

        img_name = labels_paths[idx].split('/')[-1].replace('json', 'png').replace('polygons', 'labelIds')
        img_name = save_dir + img_name
        result.save(img_name)
        # plt.imshow(result)
        # plt.show()

    print('time elapsed: {:.3f}'.format(time.time() - start))
    print('total test loss: ' + str(test_loss / len(test_loader)))
    del model


if __name__ == '__main__':
    os.environ['CITYSCAPES_RESULTS'] = '../../results'
    os.environ['CITYSCAPES_DATASET'] = '../data/raw'
    torch.cuda.empty_cache()

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='autoencoder', type=str)
    args = parser.parse_args()
    print(args)

    run(args)

