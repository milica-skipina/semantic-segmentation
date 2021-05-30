from data_loader.data_loader import Dataset
import os
import argparse
from model.autoencoder import AutoEncoder
from torch import nn
import torch
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = '../'


def prepare_data():
    # TODO load images and return data loaders
    return None, None


def capture_snapshot(dir, img, noise, output, epoch):
    """Captures and saves checkpoint model output images during training

    Args:
        dir: directory where image should be saved
        img: input image
        output: output
        noise: noise
        epoch: epoch

    Returns:
        Figure containing ground truth and predicted images
    """

    predictions = output.data.cpu().numpy().astype(float)
    for idx, prediction in enumerate(predictions):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img[idx])
        ax2.imshow(noise[idx])
        ax3.imshow(prediction)
        plt.savefig(dir + '/{}_{}.png'.format(epoch, idx))
        plt.close()
        #plt.show()
        return fig


def init_model():
    net = AutoEncoder()
    net = nn.DataParallel(net)
    net.to(DEVICE)
    print('Model loaded.')
    print('Number of model parameters:\t{}'.format(sum([p.data.nelement() for p in net.parameters()])))

    return net


def run():
    model = init_model()
    train_loader, val_loader = prepare_data()

    start = time.time()

    dtime = datetime.now().strftime("%d_%m_%H_%M")
    save_dir = ROOT + 'reports/' + dtime
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.mkdir(save_dir + '/models')
        os.mkdir(save_dir + '/images')

    writer = SummaryWriter()
    writer.add_text('hyperparameters/', str(vars(args)))

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # TODO scheduler if needed
    criterion = MSELoss()

    min_train_loss = np.inf

    print('Training loop started.')
    for epoch in range(args.epochs):
        model.train()

        train_loss = 0
        val_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        start_time = time.time()

        for batch_idx, (image, gt) in pbar:
            img = img.to(DEVICE)
            img = img.view(img.size(0), -1)
            noise = nn.Dropout(torch.ones(img.shape)).to(DEVICE)
            img_noise = (img * noise).to(DEVICE)

            prepare_time = start_time - time.time()
            # FORWARD
            output = model(img_noise)
            loss = criterion(output, img.data)
            # BACKWARD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            process_time = start_time - time.time()
            compute_efficiency = process_time/(process_time+prepare_time)
            pbar.set_description(
                f'train loss: {train_loss/(batch_idx+1):.4f}, '
                f'epoch: {epoch}/{args.epochs}: '
                f'compute efficiency: {compute_efficiency:.2f}, '
            )

            start_time = time.time()

        writer.add_scalar('loss/train_loss', train_loss/len(train_loader), epoch)

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            figure = capture_snapshot(save_dir + '/images', img, img_noise, output, epoch)
            writer.add_figure('figure/min_train_loss', figure, epoch)

    print('time elapsed: {:.3f}'.format(time.time() - start))
    del model
    writer.close()


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--optimizer-step-size', type=int, default=10)
    parser.add_argument('--optimizer-gamma', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    run(args)