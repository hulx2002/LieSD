import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import numpy as np
import torch

def visualization_vector(vector, n_visual, vis_vec_path):
    n = vector.shape[0]
    n_visual = n if n_visual > n else n_visual
    plt.plot(range(n - n_visual + 1, n + 1), vector[n - n_visual : n].numpy())
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular Value')
    plt.grid(True)
    plt.savefig(vis_vec_path, bbox_inches='tight')
    plt.clf()
    # print(vector[n - n_visual : n])

def visualization_matrix(matrices, n_visual, vis_mat_path):
    if len(matrices.shape) == 2:
        matrices.unsqueeze_(0)
    n = matrices.shape[0]
    n_visual = n if n_visual > n else n_visual
    r = round(math.sqrt(n_visual))
    c = math.ceil(n_visual / r)
    for i in range(n_visual):
        matrix = matrices[n - n_visual + i]
        plt.subplot(r, c, i + 1)
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Basis {n - n_visual + i + 1}')
        plt.xticks([])
        plt.yticks([])

    plt.savefig(vis_mat_path, bbox_inches='tight')
    plt.clf()
    # print(matrices[n - n_visual : n])

def visualization_mnist(vector, matrices, n_visual, vis_path):
    if len(matrices.shape) == 2:
        matrices.unsqueeze_(0)
    n = matrices.shape[0]
    n_visual = n if n_visual > n else n_visual
    r = round(math.sqrt(n_visual + 1))
    c = math.ceil((n_visual + 1) / r)

    plt.subplot(r, c, 1)
    plt.plot(range(1, 5), vector.numpy() / 10000)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular Value')
    plt.grid(True)

    for i in range(n_visual):
        matrix = matrices[n - n_visual + i]
        plt.subplot(r, c, i + 2)
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Basis {n - n_visual + i + 1}')
        plt.xticks([])
        plt.yticks([])

    plt.savefig(vis_path, bbox_inches='tight')
    plt.clf()
    # print(matrices[n - n_visual : n])

def evaluate(bases_x, real_bases_x, bases_y=None, real_bases_y=None):
    D = bases_x.shape[0]
    real_D = real_bases_x.shape[0]
    n = bases_x.shape[1]
    real_v_x = real_bases_x.reshape(real_D, n * n).T
    v_x = bases_x.reshape(D, n * n).T
    if bases_y is None:
        real_v = real_v_x
        v = v_x
    else:
        m = bases_y.shape[1]
        real_v_y = real_bases_y.reshape(real_D, m * m).T
        v_y = bases_y.reshape(D, m * m).T
        real_v = torch.cat((real_v_x, real_v_y), dim=0)
        v = torch.cat((v_x, v_y), dim=0)

    error_orth = 0.
    for i in range(D - 1):
        for j in range(i + 1, D):
            error_orth += torch.abs(torch.dot(v[:, i], v[:, j]))

    _, residuals_x, _, _ = np.linalg.lstsq(real_v.numpy(), v.numpy(), rcond=None)
    _, residuals_y, _, _ = np.linalg.lstsq(v.numpy(), real_v.numpy(), rcond=None)
    error_space = np.sum(residuals_x) + np.sum(residuals_y)

    return error_space, error_orth

def evaluate_2body(bases_x):
    real_bases_x = torch.zeros(1, 8, 8)
    real_bases_x[0, 0, 1] = real_bases_x[0, 2, 3] = real_bases_x[0, 4, 5] = real_bases_x[0, 6, 7] = -1.
    real_bases_x[0, 1, 0] = real_bases_x[0, 3, 2] = real_bases_x[0, 5, 4] = real_bases_x[0, 7, 6] = 1.

    bases_x_norm = F.normalize(bases_x, p=2, dim=(1, 2))
    real_bases_x_norm = F.normalize(real_bases_x, p=2, dim=(1, 2))

    error_space, error_orth = evaluate(bases_x=bases_x_norm, real_bases_x=real_bases_x_norm, bases_y=None, real_bases_y=None)
    print(f'Error_space: {error_space}')
    print(f'Error_orth: {error_orth}')

def evaluate_inertia_tensor(bases_x, bases_y_1, bases_y_2):
    real_bases_x = torch.zeros(4, 3, 3)
    real_bases_x[0, 0, 1] = real_bases_x[1, 0, 2] = real_bases_x[2, 1, 2] = real_bases_x[3, 0, 0] = real_bases_x[3, 1, 1] = real_bases_x[3, 2, 2] = 1.
    real_bases_x[0, 1, 0] = real_bases_x[1, 2, 0] = real_bases_x[2, 2, 1] = -1.
    real_bases_y_1 = real_bases_x.clone()
    real_bases_y_2 = real_bases_x.clone()

    norm = torch.sqrt(torch.norm(bases_x, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(bases_y_1, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(bases_y_2, p=2, dim=(1, 2), keepdim=True) ** 2)
    bases_x_norm = bases_x / norm
    bases_y_1_norm = bases_y_1 / norm
    bases_y_2_norm = bases_y_2 / norm
    real_norm = torch.sqrt(torch.norm(real_bases_x, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(real_bases_y_1, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(real_bases_y_2, p=2, dim=(1, 2), keepdim=True) ** 2)
    real_bases_x_norm = real_bases_x / real_norm
    real_bases_y_1_norm = real_bases_y_1 / real_norm
    real_bases_y_2_norm = real_bases_y_2 / real_norm

    bases_y = torch.zeros(bases_y_1.shape[0], bases_y_1.shape[1] * bases_y_2.shape[1], bases_y_1.shape[1] * bases_y_2.shape[1])
    for i in range(bases_y_1.shape[0]):
        bases_y[i] = torch.kron(bases_y_1_norm[i], torch.eye(bases_y_2.shape[1])) + torch.kron(torch.eye(bases_y_1.shape[1]), bases_y_2_norm[i])
    real_bases_y = torch.zeros(real_bases_y_1.shape[0], real_bases_y_1.shape[1] * real_bases_y_2.shape[1], real_bases_y_1.shape[1] * real_bases_y_2.shape[1])
    for i in range(real_bases_y_1.shape[0]):
        real_bases_y[i] = torch.kron(real_bases_y_1_norm[i], torch.eye(real_bases_y_2.shape[1])) + torch.kron(torch.eye(real_bases_y_1.shape[1]), real_bases_y_2_norm[i])
    
    error_space, error_orth = evaluate(bases_x=bases_x_norm, real_bases_x=real_bases_x_norm, bases_y=bases_y, real_bases_y=real_bases_y)
    print(f'Error_space: {error_space}')
    print(f'Error_orth: {error_orth}')

def evaluate_inertia_vector(bases_x, bases_y):
    real_bases_x = torch.zeros(4, 3, 3)
    real_bases_x[0, 0, 1] = real_bases_x[1, 0, 2] = real_bases_x[2, 1, 2] = real_bases_x[3, 0, 0] = real_bases_x[3, 1, 1] = real_bases_x[3, 2, 2] = 1.
    real_bases_x[0, 1, 0] = real_bases_x[1, 2, 0] = real_bases_x[2, 2, 1] = -1.
    real_bases_y_1 = real_bases_x.clone()
    real_bases_y_2 = real_bases_x.clone()

    norm = torch.sqrt(torch.norm(bases_x, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(bases_y, p=2, dim=(1, 2), keepdim=True) ** 2)
    bases_x_norm = bases_x / norm
    bases_y_norm = bases_y / norm
    real_norm = torch.sqrt(torch.norm(real_bases_x, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(real_bases_y_1, p=2, dim=(1, 2), keepdim=True) ** 2 + torch.norm(real_bases_y_2, p=2, dim=(1, 2), keepdim=True) ** 2)
    real_bases_x_norm = real_bases_x / real_norm
    real_bases_y_1_norm = real_bases_y_1 / real_norm
    real_bases_y_2_norm = real_bases_y_2 / real_norm

    real_bases_y = torch.zeros(real_bases_y_1.shape[0], real_bases_y_1.shape[1] * real_bases_y_2.shape[1], real_bases_y_1.shape[1] * real_bases_y_2.shape[1])
    for i in range(real_bases_y_1.shape[0]):
        real_bases_y[i] = torch.kron(real_bases_y_1_norm[i], torch.eye(real_bases_y_2.shape[1])) + torch.kron(torch.eye(real_bases_y_1.shape[1]), real_bases_y_2_norm[i])
    
    error_space, error_orth = evaluate(bases_x=bases_x_norm, real_bases_x=real_bases_x_norm, bases_y=bases_y_norm, real_bases_y=real_bases_y)
    print(f'Error_space: {error_space}')
    print(f'Error_orth: {error_orth}')

def evaluate_top_quark_tagging(bases_x):
    real_bases_x = torch.zeros(7, 4, 4)
    real_bases_x[0, 2, 1] = real_bases_x[1, 3, 1] = real_bases_x[2, 3, 2] = real_bases_x[3, 0, 1] = real_bases_x[3, 1, 0] = real_bases_x[4, 0, 2] = real_bases_x[4, 2, 0] = real_bases_x[5, 0, 3] = real_bases_x[5, 3, 0] = real_bases_x[6, 0, 0] = real_bases_x[6, 1, 1] = real_bases_x[6, 2, 2] = real_bases_x[6, 3, 3] = 1.
    real_bases_x[0, 1, 2] = real_bases_x[1, 1, 3] = real_bases_x[2, 2, 3] = -1.

    bases_x_norm = F.normalize(bases_x, p=2, dim=(1, 2))
    real_bases_x_norm = F.normalize(real_bases_x, p=2, dim=(1, 2))

    error_space, error_orth = evaluate(bases_x=bases_x_norm, real_bases_x=real_bases_x_norm, bases_y=None, real_bases_y=None)
    print(f'Error_space: {error_space}')
    print(f'Error_orth: {error_orth}')

def evaluate_mnist(bases_x):
    real_bases_x = torch.zeros(1, 2, 2)
    real_bases_x[0, 0, 1] = -1.
    real_bases_x[0, 1, 0] = 1.

    bases_x_norm = F.normalize(bases_x, p=2, dim=(1, 2))
    real_bases_x_norm = F.normalize(real_bases_x, p=2, dim=(1, 2))

    error_space, error_orth = evaluate(bases_x=bases_x_norm, real_bases_x=real_bases_x_norm, bases_y=None, real_bases_y=None)
    print(f'Error_space: {error_space}')
    print(f'Error_orth: {error_orth}')