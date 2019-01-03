import argparse
import os
from torch.backends import cudnn
from dataloader import get_loader
from solver import Solver

def main(config):
    cudnn.benchmark = True

    if not os.path.exists(config.main_path):
      os.makedirs(config.main_path)
    if not os.path.exists(os.path.join(config.main_path, 'checkpoints')):
      os.makedirs(os.path.join(config.main_path, 'checkpoints'))
    if not os.path.exists(os.path.join(config.main_path, 'samples')):
      os.makedirs(os.path.join(config.main_path, 'samples'))
    if not os.path.exists(os.path.join(config.main_path, 'logs')):
      os.makedirs(os.path.join(config.main_path, 'logs'))

    loader, data_size = get_loader(config)
    config.data_size = data_size

    solver = Solver(config, loader)
    if config.mode == 'train':
      solver.train()
    else:
      solver.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_type', type=str, default='BCEwL', help="BCEwL|WGAN|WGAN+")
    parser.add_argument('--G_lr', type=float, default=0.0002)
    parser.add_argument('--D_lr', type=float, default=0.0002)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=600)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--lr_update_step', type=int, default=14000)
    parser.add_argument('--lr_decay_after', type=int, default=14000)
    parser.add_argument('--lr_decay', type=int, default=0.5)
    parser.add_argument('--log_step', type=int, default=20)
    parser.add_argument('--eval_step', type=int, default=700) #per 5 epoch
    parser.add_argument('--save_step', type=int, default=7000) # per 50 epoch
    parser.add_argument('--main_path', type=str, default='txt2img')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--data_path', type=str, default='/home/cvpr19/scottreed/DATA/CUB_EMBEDDING/birds')
    parser.add_argument('--image_path', type=str, default='/home/cvpr19/scottreed/DATA/CUB_200_2011/images')
    parser.add_argument('--num_workers', type=int, default='4')

    config = parser.parse_args()
    print(config)
    main(config)
