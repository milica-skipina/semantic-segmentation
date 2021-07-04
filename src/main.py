import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from data_loader.cityscapes_dataset import CityscapesDataset
from model.resnet import ResNet50
from model.ae import ConvAutoencoder
# seeds
from src.model.inception import InceptionV3
from src.model.ae import ConvAutoencoder
from src.model.semanticSegmentation import SemanticSegmentationModel

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = '../'


def prepare_data():

    train_dataset = CityscapesDataset('../data/raw/leftImg8bit/train', '../data/raw/gtFine/train', dataset_len=512, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(CityscapesDataset('../data/raw/leftImg8bit/val', '../data/raw/gtFine/val', dataset_len=10, transform=True),
                            batch_size=2, shuffle=True, pin_memory=True, drop_last=False
                             )

    return train_loader, test_loader


def to_img(y):
    y = y.view(3, 256, 256)
    return y


def capture_snapshot(dir, img, output, epoch):
    """Captures and saves checkpoint model output images during training

    Args:
        dir: directory where image should be saved
        img: input image
        output: output
        epoch: epoch

    Returns:
        Figure containing ground truth and predicted images
    """

    for idx, prediction in enumerate(output):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(transforms.ToPILImage()(to_img(img[idx].cpu().data)))
        ax2.imshow(transforms.ToPILImage()(to_img(output[idx].cpu().data)))
        plt.savefig(dir + '/{}_{}.png'.format(epoch, idx))
        plt.close()
        #plt.show()
        return fig


def init_model(is_autoencoder=False):
    #net = AutoEncoder(256, 256, 400)
    #
    if is_autoencoder:
        #net = ResNet50(out_classes=3, is_autoencoder=True)
        net = ConvAutoencoder()
    else:
        # encoder = ConvAutoencoder()
        # encoder = nn.DataParallel(encoder)
        # # encoder.to(DEVICE)
        # checkpoint_path = "/home/milica/Desktop/NN/reports/19_06_15_44/autoencoderCheckpoint.pth"
        # # try:
        # loaded_checkpoint = torch.load(checkpoint_path)
        # encoder.load_state_dict(loaded_checkpoint['model_state'])
        #
        # # except:
        # #     print("Encoder not found")
        # net = SemanticSegmentationModel(encoder.module.encoder)
        # # net = MyResnetEncoder()
        # # net = ResnetEncoder(256)
        # net = nn.DataParallel(net)
        # net.to(DEVICE)
        #
        # print('Model loaded.')
        # # net = ResNet50(out_classes=8, is_autoencoder=False)
        # # [2, 1024, 16, 16]
        # # [2, 512, 13, 13]
        # return net
        # net = ResNet50(out_classes=8, is_autoencoder=False)
        net = InceptionV3(out_classes=8)
    #net = ResUNetSimple(in_channels=3, out_classes=3)
    net = nn.DataParallel(net)
    net.to(DEVICE)
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
    model = init_model(args.is_autoencoder)
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
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.optimizer_step_size, gamma=args.optimizer_gamma)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=50, min_lr=1e-7)

    if args.is_autoencoder:
        #criterion = VAELoss()
        criterion = nn.MSELoss()
    else:
        #weight = torch.tensor([0.4, 0.4, 1, 1, 1])
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    min_val_loss = np.inf
    min_train_loss = np.inf

    print('Training loop started.')
    for epoch in range(args.epochs):
        model.train()

        train_loss = 0
        val_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        start_time = time.time()

        for batch_idx, (image, gt) in pbar:
            img = image.to(DEVICE)
            gt = gt.to(DEVICE)

            prepare_time = start_time - time.time()
            # FORWARD
            if args.is_autoencoder:
                #output, mu, logvar = model(img)
                #torchvision.utils.save_image(output.data, f'{batch_idx}.png', nrow=args.batch_size, padding=2)
                #loss = criterion(img, output, mu, logvar)
                output = model(img)
                loss = criterion(img, output)
            else:
                output = model(img)
                loss = criterion(output, gt.squeeze(dim=1))

            # BACKWARD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            process_time = start_time - time.time()
            compute_efficiency = process_time/(process_time+prepare_time)
            pbar.set_description(
                f'epoch: {epoch}/{args.epochs}: '
                f'train loss: {train_loss/(batch_idx+1):.4f}, '
                f'compute efficiency: {compute_efficiency:.2f}, '
                f'lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.8f}, '
            )

            start_time = time.time()

        writer.add_scalar('loss/train_loss', train_loss/len(train_loader), epoch)

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            if args.is_autoencoder:
                predictions = output
            else:
                output = F.log_softmax(output, dim=1)
                predictions = torch.argmax(output, axis=1).data.cpu().numpy()

            for idx, prediction in enumerate(predictions):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                ax1.imshow(img[idx].data.cpu().numpy().transpose(1, 2, 0))
                ax2.imshow(gt[idx].data.cpu().numpy())
                if args.is_autoencoder:
                    ax3.imshow(predictions[idx].data.cpu().numpy().transpose(1, 2, 0))
                else:
                    ax3.imshow(predictions[idx])
                plt.savefig(save_dir + '/images' + '/{}_{}.png'.format(epoch, idx))
                plt.close()
                # plt.show()
                figure = fig
            writer.add_figure('figure/min_train_loss', figure, epoch)
            #torch.save(model.module.state_dict(), save_dir + '/models/' + str(epoch) + '_best_ae_ever.pth')
            '''checkpoint_path = save_dir + "/models/decoderCheckpoint.pth"
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)'''
        if epoch % 20 == 0:
            model.eval()
            pbar = tqdm(enumerate(val_loader), total=len(val_loader))

            with torch.no_grad():
                for batch_idx, (image, gt) in pbar:
                    img = image.to(DEVICE)
                    gt = gt.to(DEVICE)

                    if args.is_autoencoder:
                        # output, mu, logvar = model(img)
                        # loss = criterion(img, output, mu, logvar)
                        output = model(img)
                        loss = criterion(img, output)
                    else:
                        output = model(img)
                        loss = criterion(output, gt.squeeze(dim=1))

                    val_loss += loss.item()
                    pbar.set_description(
                        f'val loss: {val_loss / (batch_idx + 1):.4f}, '
                        f'epoch: {epoch}/{args.epochs}: ')

            writer.add_scalar('loss/val_loss', val_loss / len(val_loader), epoch)
            if val_loss < min_val_loss:
                min_val_loss = val_loss

                if args.is_autoencoder:
                    predictions = output
                else:
                    output = F.log_softmax(output, dim=1)
                    predictions = torch.argmax(output, axis=1).data.cpu().numpy()

                for idx, prediction in enumerate(predictions):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                    ax1.imshow(img[idx].data.cpu().numpy().transpose(1, 2, 0))
                    ax2.imshow(gt[idx].data.cpu().numpy())
                    if args.is_autoencoder:
                        ax3.imshow(predictions[idx].data.cpu().numpy().transpose(1, 2, 0))
                    else:
                        ax3.imshow(predictions[idx])
                    plt.savefig(save_dir + '/images' + '/{}_{}.png'.format(epoch, idx))
                    plt.close()
                    # plt.show()
                    figure = fig
                writer.add_figure('figure/min_val_loss', figure, epoch)

                # if epoch > 1:
                    # torch.save(model.module.state_dict(), save_dir + '/models/' + str(epoch) + '_segnet.pth')

        lr_scheduler.step(train_loss)
        print(f'\nEpoch {epoch}/{args.epochs}, '
              f'training loss: {train_loss / len(train_loader)}, '
              f'validation loss: {val_loss / len(val_loader)}\n'
              f'lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.8f}, ')


        print('time elapsed: {:.3f}'.format(time.time() - start))
    del model
    writer.close()


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--optimizer-step-size', type=int, default=30)
    parser.add_argument('--optimizer-gamma', type=float, default=0.5)
    parser.add_argument('--is-autoencoder', default=True, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    args = parser.parse_args()
    print(args)

    run(args)
