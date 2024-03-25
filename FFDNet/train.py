import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import logging
from models import FFDNet
from utils import weights_init_kaiming, batch_psnr, svd_orthogonalization
import lmdb
from PIL import Image
from io import BytesIO
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'training.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler()])

def read_images_from_lmdb(lmdb_path):
    images = []
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            buf = BytesIO(value)
            img = Image.open(buf).convert('RGB')
            images.append(img)
    env.close()
    return images
import torch
import numpy as np
from torch.utils.data import Dataset

def extract_patches(images, patch_size, n_patches):
    """Extracts n_patches patches of size patch_size from the input images."""
    patches = []
    # Calculate the number of patches to extract along the height and width
    h_patches = w_patches = int(np.sqrt(n_patches))
    patch_height, patch_width = patch_size, patch_size
    step_h, step_w = images.shape[1] // h_patches, images.shape[2] // w_patches
    for i in range(h_patches):
        for j in range(w_patches):
            patch = images[:, 
                           i * step_h : i * step_h + patch_height, 
                           j * step_w : j * step_w + patch_width]
            patches.append(patch)
    return patches

class SIDDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, patch_size=128, n_patches=16):
        self.images_gt = read_images_from_lmdb(gt_dir)
        self.images_noisy = read_images_from_lmdb(noisy_dir)
        self.patch_size = patch_size
        self.n_patches = n_patches

    def __len__(self):
        return len(self.images_gt) * self.n_patches  # Now each image results in multiple patches

    def __getitem__(self, idx):
        image_idx = idx // self.n_patches  # Find out which image this index corresponds to
        patch_idx = idx % self.n_patches  # Find out which patch of the image this corresponds to

        gt_image = self.images_gt[image_idx]
        noisy_image = self.images_noisy[image_idx]

        gt_image = torch.FloatTensor(np.array(gt_image) / 255.0)
        noisy_image = torch.FloatTensor(np.array(noisy_image) / 255.0)

        gt_image = gt_image.permute(2, 0, 1)
        noisy_image = noisy_image.permute(2, 0, 1)

        # Extract patches
        gt_patches = extract_patches(gt_image, self.patch_size, self.n_patches)
        noisy_patches = extract_patches(noisy_image, self.patch_size, self.n_patches)

        return {'gt': gt_patches[patch_idx], 'noisy': noisy_patches[patch_idx]}


def main(args):
    setup_logging(args.log_dir)
    logging.info('> Loading dataset ...')
    dataset_train = SIDDataset(gt_dir='../datasets/SIDD/train/gt_crops.lmdb', noisy_dir='../datasets/SIDD/train/input_crops.lmdb', patch_size=128, n_patches=16)
    dataset_val = SIDDataset(gt_dir='../datasets/SIDD/val/gt_crops.lmdb', noisy_dir='../datasets/SIDD/val/input_crops.lmdb', patch_size=128, n_patches=4)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=args.batch_size, shuffle=False)
    logging.info("\t# of training samples: %d" % int(len(dataset_train)))

    net = FFDNet(num_input_channels=3)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    model = nn.DataParallel(net, device_ids=[0]).cuda()
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_psnr_val = 0

    if args.resume_training:
        resumef = os.path.join(args.log_dir, 'ckpt.pth')
        if os.path.isfile(resumef):
            checkpoint = torch.load(resumef)
            logging.info("> Resuming previous training")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args = checkpoint['args']
            start_epoch = checkpoint['epoch']
            logging.info("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
        else:
            logging.error("No checkpoint found at '{}'!".format(resumef))
            return
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, data in enumerate(loader_train):
            img_train, imgn_train = data['gt'], data['noisy']
            img_train, imgn_train = img_train.cuda(), imgn_train.cuda()
            noise = imgn_train - img_train
            img_train, imgn_train, noise = Variable(img_train), Variable(imgn_train), Variable(noise)
            stdn_var = torch.std(noise, dim=(1, 2, 3))

            optimizer.zero_grad()
            out_train = model(imgn_train, stdn_var)
            loss = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            loss.backward()
            optimizer.step()

            if i % args.save_every == 0:
                logging.info("[epoch %d][%d/%d] loss: %.4f" %
                             (epoch + 1, i + 1, len(loader_train), loss.item()))

        if (epoch + 1) % args.save_every_epochs == 0:
            save_path = os.path.join(args.log_dir, f'ckpt_epoch_{epoch + 1}.pth')
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': args}, save_path)
            logging.info(f"Checkpoint saved to {save_path}")

        model.eval()
        psnr_val = 0
        for data in loader_val:
            img_val, imgn_val = data['gt'], data['noisy']
            img_val, imgn_val = img_val.cuda(), imgn_val.cuda()
            noise_val = imgn_val - img_val
            stdn_var_val = torch.std(noise_val, dim=(1, 2, 3))
            out_val = torch.clamp(imgn_val - model(imgn_val, stdn_var_val), 0., 1.)
            psnr_val += batch_psnr(out_val, img_val, 1.)
        psnr_val /= len(loader_val)
        if psnr_val > best_psnr_val :
            best_psnr_val = psnr_val
            save_path = os.path.join(args.log_dir, f'ckpt_best_psnr.pth')
            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': args}, save_path)
            logging.info(f"Checkpoint saved to {save_path}")
            
        logging.info("[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFDNet")
    parser.add_argument("--log_dir", type=str, default="logs", help='path of log files')
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", "--e", type=int, default=80, help="Number of total training epochs")
    parser.add_argument("--resume_training", "--r", action='store_true', help="resume training from a previous checkpoint")
    parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--no_orthog", action='store_true', help="Don't perform orthogonalization as regularization")
    parser.add_argument("--save_every", type=int, default=200, help="Number of training steps to log psnr and save model")
    parser.add_argument("--save_every_epochs", type=int, default=20, help="Number of training epochs to save state")
    args = parser.parse_args()

    main(args)
