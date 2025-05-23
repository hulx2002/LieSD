# code from https://github.com/Rose-STL-Lab/LieGAN/blob/master/main_lagan.py

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import *
from gan import LieGenerator, LieDiscriminator, LieDiscriminatorEmb
from train import train_lie_gan, train_lie_gan_incremental
from visualization import *
from torchvision import datasets, transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model & training settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr_d', type=float, default=2e-4)
    parser.add_argument('--lr_g', type=float, default=1e-3)
    parser.add_argument('--reg_type', type=str, default='cosine')
    parser.add_argument('--lamda', type=float, default=1e-2)
    parser.add_argument('--p_norm', type=float, default=2)
    parser.add_argument('--droprate_init', type=float, default=0.8)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--activate_threshold', action='store_true')
    parser.add_argument('--D_loss_threshold', type=float, default=0.25)
    parser.add_argument('--model', type=str, default='lie')
    parser.add_argument('--coef_dist', type=str, default='normal')
    parser.add_argument('--g_init', type=str, default='random')
    parser.add_argument('--sigma_init', type=float, default=1)
    parser.add_argument('--uniform_max', type=float, default=1)
    parser.add_argument('--normalize_Li', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--n_channel', type=int, default=1)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--incremental', action='store_true')
    # dataset settings
    parser.add_argument('--task', type=str, default='traj_pred')
    parser.add_argument('--dataset_name', type=str, default='2body')
    parser.add_argument('--dataset_config', type=str, nargs='+', default=None)
    parser.add_argument('--dataset_size', type=int, default=2000)
    parser.add_argument('--x_type', type=str, default='vector')
    parser.add_argument('--y_type', type=str, default='vector')
    parser.add_argument('--input_timesteps', type=int, default=1)
    parser.add_argument('--output_timesteps', type=int, default=1)
    parser.add_argument('--n_component', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--non_uniform', action='store_true')
    parser.add_argument('--degree', type=float, default=0.0)
    # run settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='saved_model')
    parser.add_argument('--save_name', type=str, default='default')
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    if args.task == 'traj_pred':
        dataset = NBodyDataset(
            input_timesteps=args.input_timesteps,
            output_timesteps=args.output_timesteps,
            save_path=f'../data/hnn/{args.dataset_name}-orbits-dataset.pkl',
            extra_features=args.dataset_config,
            non_uniform=args.non_uniform,
        )
        if args.dataset_config is None:
            n_dim = 8
        elif 'log' in args.dataset_config:
            n_dim = 5
        n_channel = args.n_channel
        d_input_size = n_dim * (args.input_timesteps + args.output_timesteps)
    elif args.task == 'top_tagging':
        dataset = TopQuarkTagging(
            path='../data/top-quark-tagging/train.h5',
            n_component=args.n_component,
            noise=args.noise,
        )
        n_dim = 4
        n_channel = args.n_channel
        n_component = args.n_component
        d_input_size = n_dim * n_component
        n_class = 2
        emb_size = 32
    elif args.task == 'mnist':
        transform = transforms.Compose([
            transforms.RandomRotation(args.degree),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = datasets.MNIST(root='../data', train=True, download=False, transform=transform)
        n_dim = 2
        n_channel = 1
        d_input_size = 28 * 28
        n_class = 10
        emb_size = 32
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize generator and discriminator
    if args.model in ['lie', 'lie_subgrp']:
        generator = LieGenerator(n_dim, n_channel, args).to(args.device)
    if args.task in ['top_tagging', 'mnist']:
        discriminator = LieDiscriminatorEmb(d_input_size, n_class, emb_size).to(args.device)
    else:
        discriminator = LieDiscriminator(d_input_size).to(args.device)
    if args.model == 'lie':  # fix the coefficient distribution
        generator.mu.requires_grad = False
        generator.sigma.requires_grad = False
    elif args.model == 'lie_subgrp':  # fix the generator
        generator.Li.requires_grad = False

    # Train
    train_fn = train_lie_gan if not args.incremental else train_lie_gan_incremental
    train_fn(
        generator,
        discriminator,
        dataloader,
        args.num_epochs,
        args.lr_d,
        args.lr_g,
        args.reg_type,
        args.lamda,
        args.p_norm,
        args.mu,
        args.eta,
        args.device,
        task=args.task,
        save_path=f'{args.save_path}/{args.save_name}/',
        print_every=args.print_every,
    )

    if args.task == 'traj_pred':
        temp = 'non_uniform' if args.non_uniform else 'uniform'
        visualization_matrix(generator.getLi().detach().cpu(), args.n_channel, f'result/{args.model}_2body_{temp}_{args.seed}seed.png')
        evaluate_2body(generator.getLi().detach().cpu())
    elif args.task == 'top_tagging':
        visualization_matrix(generator.getLi().detach().cpu(), args.n_channel, f'result/{args.model}_top_{args.n_channel}channels_{args.noise}noise_{args.seed}seed.png')
        evaluate_top_quark_tagging(generator.getLi().detach().cpu())
    elif args.task == 'mnist':
        visualization_matrix(generator.getLi().detach().cpu(), args.n_channel, f'result/{args.model}_mnist_{int(args.degree)}degrees_{args.seed}seed.png')
        evaluate_mnist(generator.getLi().detach().cpu())