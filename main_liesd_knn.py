import argparse
from torch.utils.data import DataLoader
from dataset import *
from discovery_knn import *
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
        discovery_batch_size = len(dataset)
    elif args.dataset == 'inertia':
        dataset = Inertia(
            N = args.N,
            k = args.k,
            noise = args.noise,
            tensor=args.tensor,
        )
        discovery_batch_size = len(dataset)

    discovery_dataloader = DataLoader(dataset, discovery_batch_size, shuffle=False)

    if args.dataset == '2body':
        A, sigma = discovery_2body_knn(discovery_dataloader, args.n_neighbors)
    elif args.dataset == 'inertia':
        if args.tensor:
            Ax, Ay1, Ay2, sigma = discovery_inertia_tensor(discovery_dataloader, args.n_neighbors)
        else:
            Ax, Ay, sigma = discovery_inertia_vector(discovery_dataloader, args.device)

    if args.dataset == '2body':
        temp = 'non_uniform' if args.non_uniform else 'uniform'
        visualization_vector(sigma, 6 * args.D, f'result/sv_2body_{temp}_knn.png')
        visualization_matrix(A, args.D, f'result/basis_2body_{temp}_knn.png')
        evaluate_2body(A[A.shape[0] - 1 :])
    elif args.dataset == 'inertia':
        if args.tensor:
            visualization_vector(sigma, sigma.shape[0], f'result/sv_inertia_tensor_{args.noise}noise_knn.png')
            visualization_matrix(Ax, args.D, f'result/basis_inertia_x_tensor_{args.noise}noise_knn.png')  
            visualization_matrix(Ay1, args.D, f'result/basis_inertia_y1_tensor_{args.noise}noise_knn.png')  
            visualization_matrix(Ay2, args.D, f'result/basis_inertia_y2_tensor_{args.noise}noise_knn.png')
            evaluate_inertia_tensor(Ax[Ax.shape[0] - 5 :], Ay1[Ay1.shape[0] - 5 :], Ay2[Ay2.shape[0] - 5 :])
        else:
            visualization_vector(sigma, sigma.shape[0], f'result/sv_inertia_vector_{args.noise}noise_knn.png')
            visualization_matrix(Ax, args.D, f'result/basis_inertia_x_vector_{args.noise}noise_knn.png')  
            visualization_matrix(Ay, args.D, f'result/basis_inertia_y_vector_{args.noise}noise_knn.png')
            evaluate_inertia_vector(Ax[Ax.shape[0] - 26 :], Ay[Ay.shape[0] - 26 :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Not fixed
    parser.add_argument('--dataset', type=str, default='2body')
    parser.add_argument('--non_uniform', action='store_true')
    parser.add_argument('--tensor', action='store_true')
    parser.add_argument('--D', type=int, default=1)
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--N', type=int, default=100000)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.0)
    # Fixed
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--input_timesteps', type=int, default=1)
    parser.add_argument('--output_timesteps', type=int, default=1)
    parser.add_argument('--dataset_config', type=str, nargs='+', default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)