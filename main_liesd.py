import argparse
from torch.utils.data import DataLoader
from dataset import *
from model import *
from train import *
from discovery import *
from visualization import *
from torchvision import datasets, transforms

def main(args):
    if args.dataset == '2body':
        dataset = NBodyDataset(
            input_timesteps=args.input_timesteps, 
            output_timesteps=args.output_timesteps,
            save_path=f'./data/hnn/2body-orbits-dataset.pkl',
            extra_features=args.dataset_config,
            multi_channel=True,
            non_uniform=args.non_uniform,
        )
        input_dim = 8
        hidden_dim = 384
        output_dim = 8
        nonlinearity = 'tanh'
        discovery_batch_size = len(dataset)
        classify = False
    elif args.dataset == 'inertia':
        dataset = Inertia(
            N = args.N,
            k = args.k,
            noise = args.noise,
            tensor=args.tensor,
        )
        input_dim = 3 * args.k
        hidden_dim = 384
        output_dim = 9
        nonlinearity = 'relu'
        discovery_batch_size = len(dataset)
        classify = False
    elif args.dataset == 'top_quark_tagging':
        dataset = TopQuarkTagging(
            n_component=args.n_component,
            noise=args.noise,
            multi_channel=True,
        )
        input_dim = 4 * args.n_component
        hidden_dim = 10 * args.n_component
        output_dim = 1
        nonlinearity = 'relu'
        discovery_batch_size = len(dataset) // 10
        classify = True
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.RandomRotation(args.degree),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        discovery_batch_size = len(dataset)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    if args.dataset == 'mnist':
        net = CNN().to(args.device)
    else:
        net = MLP(input_dim, hidden_dim, output_dim, nonlinearity, classify).to(args.device)
    train_net(net, train_dataloader, args.num_epochs, args.device, args.train_print_every, args.dataset)
    discovery_dataloader = DataLoader(dataset, discovery_batch_size, shuffle=True)

    if args.dataset == '2body':
        A, sigma = discovery_2body(net, discovery_dataloader, args.device)
    elif args.dataset == 'inertia':
        if args.tensor:
            Ax, Ay1, Ay2, sigma = discovery_inertia_tensor(net, discovery_dataloader, args.device)
        else:
            Ax, Ay, sigma = discovery_inertia_vector(net, discovery_dataloader, args.device)
    elif args.dataset == 'top_quark_tagging':
        A, sigma = discovery_top_quark_tagging(net, discovery_dataloader, args.device)
    elif args.dataset == 'mnist':
        A, sigma = discovery_mnist(net, discovery_dataloader, args.device)

    if args.dataset == '2body':
        temp = 'non_uniform' if args.non_uniform else 'uniform'
        visualization_vector(sigma, 6 * args.D, f'result/sv_2body_{temp}_{args.seed}seed.png')
        visualization_matrix(A, args.D, f'result/basis_2body_{temp}_{args.seed}seed.png')
        evaluate_2body(A[A.shape[0] - 1 :])
    elif args.dataset == 'inertia':
        if args.tensor:
            visualization_vector(sigma, sigma.shape[0], f'result/sv_inertia_tensor_{args.noise}noise_{args.seed}seed.png')
            visualization_matrix(Ax, args.D, f'result/basis_inertia_x_tensor_{args.noise}noise_{args.seed}seed.png')  
            visualization_matrix(Ay1, args.D, f'result/basis_inertia_y1_tensor_{args.noise}noise_{args.seed}seed.png')  
            visualization_matrix(Ay2, args.D, f'result/basis_inertia_y2_tensor_{args.noise}noise_{args.seed}seed.png')
            evaluate_inertia_tensor(Ax[Ax.shape[0] - 5 :], Ay1[Ay1.shape[0] - 5 :], Ay2[Ay2.shape[0] - 5 :])
        else:
            visualization_vector(sigma, sigma.shape[0], f'result/sv_inertia_vector_{args.noise}noise_{args.seed}seed.png')
            visualization_matrix(Ax, args.D, f'result/basis_inertia_x_vector_{args.noise}noise_{args.seed}seed.png')  
            visualization_matrix(Ay, args.D, f'result/basis_inertia_y_vector_{args.noise}noise_{args.seed}seed.png')
            evaluate_inertia_vector(Ax[Ax.shape[0] - 26 :], Ay[Ay.shape[0] - 26 :])
    elif args.dataset == 'top_quark_tagging':
        visualization_vector(sigma, sigma.shape[0], f'result/sv_top_{args.noise}noise_{args.seed}seed.png')
        visualization_matrix(A, args.D, f'result/basis_top_{args.noise}noise_{args.seed}seed.png')
        evaluate_top_quark_tagging(A[A.shape[0] - 7 :])
    elif args.dataset == 'mnist':
        visualization_mnist(sigma, A, args.D, f'result/mnist_{int(args.degree)}degrees_{args.seed}seed.png')
        evaluate_mnist(A[A.shape[0] - 1 :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Not fixed
    parser.add_argument('--dataset', type=str, default='2body')
    parser.add_argument('--non_uniform', action='store_true')
    parser.add_argument('--tensor', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--train_print_every', type=int, default=1)
    parser.add_argument('--D', type=int, default=1)
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n_component', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--degree', type=float, default=0.0)
    # Fixed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_timesteps', type=int, default=1)
    parser.add_argument('--output_timesteps', type=int, default=1)
    parser.add_argument('--dataset_config', type=str, nargs='+', default=None)
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)