import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Data loader.
    data_loader = None

    if config.dataset == 'rsna' or config.dataset == 'vincxr' or config.dataset == 'lag':
        data_loader = get_loader(config.dataset,config.batch_size,config.image_size,config.num_workers,config.mode)
        
    # Solver for training and testing Brainomaly.
    solver = Solver(data_loader, config)
    
    if config.mode == 'train':
        if config.dataset in ['rsna','vincxr','lag']:
            solver.train()
    elif config.mode == 'test':
        if config.dataset in ['rsna','vincxr','lag']: 
            solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--image_size', type=int, default=64, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity loss')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='rsna', choices=['rsna','vincxr','lag']) 
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=100000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=50000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.00005, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.00005, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=2, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--device',type=int,default=0,help='train on the GPU')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=100000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train','test'])

    # Directories.
    parser.add_argument('--save_dir', type=str, default='rsna/save_test')
    parser.add_argument('--model_save_dir', type=str, default='rsna/models')
    parser.add_argument('--sample_dir', type=str, default='rsna/samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--lr_update_step', type=int, default=1000)


    config = parser.parse_args()
    main(config)
