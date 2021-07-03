from data_loader.cityscapes_dataset import CityscapesDataset
import os
import argparse
from torch import nn
import torch
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from src.model.convAutoencoder import ConvAutoencoder
from src.model.resnetEncoder import MyResnetEncoder
from src.model.semanticSegmentation import SemanticSegmentationModel
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = '../'


def prepare_data():
    train_dataset = CityscapesDataset('../data/raw/leftImg8bit/train', '../data/raw/gtFine/train', dataset_len=100, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(CityscapesDataset('../data/raw/leftImg8bit/val', '../data/raw/gtFine/val', dataset_len=100, transform=True),
                            batch_size=20, shuffle=True, pin_memory=True, drop_last=False
                             )

    return train_loader, test_loader


def to_img(y):
    y = y.view(1, 256, 256)
    return y

def to_img_output(y):
    y = torch.argmax(y, dim=0)
    y = np.uint8(y)
    return y

# def cat2label(label):
#     pixels = label.load()
#     for i in range(label.size[0]):
#         for j in range(label.size[1]):
#             if pixels[i, j] == 1:
#                 pixels[i, j] = 3
#             if pixels[i, j] == 2:
#                 pixels[i, j] = 6
#             if pixels[i, j] == 3:
#                 pixels[i, j] = 7
#     return label

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


def init_model():
    encoder = ConvAutoencoder()
    encoder = nn.DataParallel(encoder)
    # encoder.to(DEVICE)
    checkpoint_path = "/home/milica/Desktop/NN/reports/19_06_15_44/autoencoderCheckpoint.pth"
    # try:
    loaded_checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(loaded_checkpoint['model_state'])

    # except:
    #     print("Encoder not found")
    net = SemanticSegmentationModel(encoder.module.encoder)
    # net = MyResnetEncoder()
    # net = ResnetEncoder(256)
    net = nn.DataParallel(net)
    net.to(DEVICE)

    print('Model loaded.')

    return net


def validate_model(model, val_loader, criterion):
    model.eval()

    val_loss = 0
    # pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    for batch_idx, (image, gt) in enumerate(val_loader):
        image = image.to(DEVICE)
        img = image
        output = model(img)

        loss = criterion(output, gt)

        val_loss += loss.item()

    return val_loss / len(val_loader)


def run(args):
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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=30, min_lr=1e-7)
    weights = torch.tensor([0.4, 0.4, 0.4, 0.5, 1, 0.5, 1, 0.7], dtype=torch.float32)
    criterion = CrossEntropyLoss()

    min_train_loss = np.inf

    # checkpoint_path = save_dir + "/decoderCheckpoint.pth"
    checkpoint_path = "/home/milica/Desktop/NN/reports/02_07_19_14/decoderCheckpoint.pth"
    try:
        loaded_checkpoint = torch.load(checkpoint_path)
        loaded_epoch = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['model_state'])
        # optimizer.load_state_dict(loaded_checkpoint['optim_state'])
        # print("FOUND")
    except:
        print("Model not found")
        loaded_epoch = 0

    print('Training loop started.')

    if loaded_epoch > 0:
        val_loss = validate_model(model, val_loader, criterion)
        min_val_loss = val_loss
        print("VALIDATION:")
        print(val_loss)
    for epoch in range(loaded_epoch, args.epochs):
        model.train()

        train_loss = 0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        start_time = time.time()

        for batch_idx, (image, gt) in pbar:
            image = image.to(DEVICE)
            img = image
            prepare_time = start_time - time.time()
            # FORWARD
            output = model(img)
            # output = output.numpy().argmax(output, axis=0)
            # temp2 = torch.clone(output)
            # temp2[:, 0, :, :] = temp2[:, 0, :, :]*0.9
            # # temp2[:, 1, :, :] = temp2[:, 0, :, :]*1.2
            # temp2[:, 2, :, :] = temp2[:, 0, :, :]*1.2
            # temp2[:, 3, :, :] = temp2[:, 0, :, :]*1.1

            # temp = torch.squeeze(gt).long()
            loss = criterion(output, gt.squeeze(dim=1))
            # BACKWARD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            scheduler.step(train_loss)

            process_time = start_time - time.time()
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f'train loss: {train_loss / (batch_idx + 1):.4f}, '
                f'epoch: {epoch}/{args.epochs}: '
                f'compute efficiency: {compute_efficiency:.2f}, '
                f'lr: {optimizer.state_dict()["param_groups"][0]["lr"]:.8f}, '

            )

            start_time = time.time()

        writer.add_scalar('loss/train_loss', train_loss / len(train_loader), epoch)

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            # val_loss = validate_model(model, val_loader, criterion)
            # if val_loss < min_val_loss:
            #     min_val_loss = val_loss
            #     print("------------------- VALIDATION -------------------")
            #     print(min_val_loss)
            if args.is_autoencoder:
                predictions = output
            else:
                output = F.log_softmax(output, dim=1)
                predictions = torch.argmax(output, axis=1).data.cpu().numpy()
            for idx, prediction in enumerate(predictions):
                if idx > 13:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                    ax1.imshow(img[idx].data.cpu().numpy().transpose(1, 2, 0))
                    ax2.imshow(gt[idx].data.cpu().numpy())
                    if args.is_autoencoder:
                        ax3.imshow(torch.sigmoid(predictions[idx]).data.cpu().numpy().transpose(1, 2, 0))
                    else:
                        ax3.imshow(predictions[idx])
                    plt.savefig(save_dir + '/images' + '/{}_{}.png'.format(epoch, idx))
                    plt.close()
                    # plt.show()
                    figure = fig
            # figure = capture_snapshot(save_dir + '/images', gt, output, epoch)
            writer.add_figure('figure/min_train_loss', figure, epoch)
            checkpoint_path = save_dir + "/decoderCheckpoint.pth"
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)

    print('time elapsed: {:.3f}'.format(time.time() - start))
    model_path = './model.pth'
    model_path_dict = './model_dict.pth'

    torch.save(model.state_dict(), model_path_dict)
    torch.save(model, model_path)
    del model
    writer.close()


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=3e-5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--optimizer-step-size', type=int, default=30)
    parser.add_argument('--optimizer-gamma', type=float, default=0.5)
    parser.add_argument('--is-autoencoder', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    args = parser.parse_args()
    print(args)

    run(args)
