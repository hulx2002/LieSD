import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

def calculate_dydx(x, y, n_neighbors):
    batch_size = x.shape[0]
    x_dim = x.shape[1]
    y_dim = y.shape[1]
    if n_neighbors == -1:
        n_neighbors = batch_size
    dydx = np.zeros((batch_size, y_dim, x_dim))
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(x)
    for i in trange(batch_size):
        neigh_dist, neigh_ind = neigh.kneighbors([x[i]])
        nearest_x = x[neigh_ind[0]]
        nearest_y = y[neigh_ind[0]]
        W = np.exp(-neigh_dist[0] ** 2 / 2)
        A = nearest_x - np.tile(x[i][None, :], (n_neighbors, 1))
        B = nearest_y - np.tile(y[i][None, :], (n_neighbors, 1))
        A = A * np.tile(W[:, None], (1, x_dim))
        B = B * np.tile(W[:, None], (1, y_dim))
        dydx[i] = np.linalg.lstsq(A, B, rcond=None)[0].T
    return dydx

def discovery_2body_knn(discovery_dataloader, n_neighbors):
    print("Start symmetry discovery.")
    for x, y in discovery_dataloader:
        batch_size = x.shape[0]
        x_channel = x.shape[1]
        x_dim = x.shape[2]
        y_channel = y.shape[1]
        y_dim = y.shape[2]
        x = x.reshape(batch_size, x_channel * x_dim)
        y = y.reshape(batch_size, y_channel * y_dim)
        dydx = torch.from_numpy(calculate_dydx(x.numpy(), y.numpy(), n_neighbors))
        CTC = torch.zeros((x_channel * x_dim * x_dim, x_channel * x_dim * x_dim))
        for i in range(batch_size):
            Cxi = torch.zeros((y_channel * y_dim, x_channel * x_dim * x_dim))
            Cyi = torch.zeros((y_channel * y_dim, y_channel * y_dim * y_dim))
            for k in range(y_channel):
                Cyi[k * y_dim : (k + 1) * y_dim, k * y_dim * y_dim : (k + 1) * y_dim * y_dim] = torch.kron(torch.eye(y_dim), y[i, k * y_dim : (k + 1) * y_dim].unsqueeze(1).T)
                for l in range(x_channel):
                    Cxi[k * y_dim : (k + 1) * y_dim, l * x_dim * x_dim : (l + 1) * x_dim * x_dim] = -torch.kron(dydx[i, k * y_dim : (k + 1) * y_dim, l * x_dim : (l + 1) * x_dim], x[i, l * x_dim : (l + 1) * x_dim].unsqueeze(1).T)
            CTC += (Cxi + Cyi).T @ (Cxi + Cyi)
        U, S, V = torch.svd(CTC)
        A = torch.zeros((x_channel * x_dim * x_dim, x_channel * x_dim, x_channel * x_dim))
        for i in range(x_channel):
            A[:, i * x_dim : (i + 1) * x_dim, i * x_dim : (i + 1) * x_dim] = V[i * x_dim * x_dim : (i + 1) * x_dim * x_dim, :].T.reshape(x_channel * x_dim * x_dim, x_dim, x_dim)
        sigma = torch.sqrt(S)
        print("End symmetry discovery.")
        return A, sigma

def discovery_inertia_tensor(discovery_dataloader, n_neighbors):
    print("Start symmetry discovery.")
    for x, y in discovery_dataloader:
        batch_size = x.shape[0]
        x_channel = x.shape[1]
        x_dim = x.shape[2]
        y_dim_1 = y.shape[1]
        y_dim_2 = y.shape[2]
        x = x.reshape(batch_size, x_channel * x_dim)
        y = y.reshape(batch_size, y_dim_1 * y_dim_2)
        dydx = torch.from_numpy(calculate_dydx(x.numpy(), y.numpy(), n_neighbors))
        CTC = torch.zeros((x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2, x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2))
        for i in range(batch_size):
            Cxi = torch.zeros((y_dim_1 * y_dim_2, x_dim * x_dim))
            for j in range(x_channel):
                Cxi -= torch.kron(dydx[i, :, j * x_dim : (j + 1) * x_dim], x[i, j * x_dim : (j + 1) * x_dim].unsqueeze(1).T)
            Yi = y[i].reshape(y_dim_1, y_dim_2)
            Cy1i = torch.kron(torch.eye(y_dim_1), Yi.T.contiguous())
            Cy2i = torch.zeros((y_dim_1 * y_dim_2, y_dim_2 * y_dim_2))
            for j in range(y_dim_1):
                Cy2i[j * y_dim_2 : (j + 1) * y_dim_2, :] = torch.kron(torch.eye(y_dim_2), Yi[j].unsqueeze(0))
            index1 = x_dim * x_dim
            index2 = x_dim * x_dim + y_dim_1 * y_dim_1
            CTC[: index1, : index1] += Cxi.T @ Cxi
            CTC[: index1, index1 : index2] += Cxi.T @ Cy1i
            CTC[: index1, index2 :] += Cxi.T @ Cy2i
            CTC[index1 : index2, : index1] += Cy1i.T @ Cxi
            CTC[index1 : index2, index1 : index2] += Cy1i.T @ Cy1i
            CTC[index1 : index2, index2 :] += Cy1i.T @ Cy2i
            CTC[index2 :, : index1] += Cy2i.T @ Cxi
            CTC[index2 :, index1 : index2] += Cy2i.T @ Cy1i
            CTC[index2 :, index2 :] += Cy2i.T @ Cy2i
        U, S, V = torch.svd(CTC)
        Vx = V[: x_dim * x_dim, :]
        Vy1 = V[x_dim * x_dim : x_dim * x_dim + y_dim_1 * y_dim_1, :]
        Vy2 = V[x_dim * x_dim + y_dim_1 * y_dim_1 :, :]
        Ax = Vx.T.reshape(x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2, x_dim, x_dim)
        Ay1 = Vy1.T.reshape(x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2, y_dim_1, y_dim_1)
        Ay2 = Vy2.T.reshape(x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2, y_dim_2, y_dim_2)
        sigma = torch.sqrt(S)
        print("End symmetry discovery.")
        return Ax, Ay1, Ay2, sigma
