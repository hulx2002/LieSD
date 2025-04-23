import torch
from tqdm import trange

def discovery_2body(net, discovery_dataloader, device):
    # for param in net.parameters():
        # param.requires_grad = False
    print("Start symmetry discovery.")
    for x, y in discovery_dataloader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]
        x_channel = x.shape[1]
        x_dim = x.shape[2]
        y_channel = y.shape[1]
        y_dim = y.shape[2]
        x = x.reshape(batch_size, x_channel * x_dim)
        y = y.reshape(batch_size, y_channel * y_dim)
        CTC = torch.zeros((x_channel * x_dim * x_dim, x_channel * x_dim * x_dim), device=device)
        for i in trange(batch_size):
            xi = x[i, :]
            xi.requires_grad = True
            fi = net(xi)
            dfi = torch.zeros((y_channel * y_dim, x_channel * x_dim), device=device)
            for j in range(y_channel * y_dim):
                dfi[j, :] = torch.autograd.grad(fi[j], xi, retain_graph=True)[0]
            Cxi = torch.zeros((y_channel * y_dim, x_channel * x_dim * x_dim), device=device)
            Cyi = torch.zeros((y_channel * y_dim, y_channel * y_dim * y_dim), device=device)
            for k in range(y_channel):
                Cyi[k * y_dim : (k + 1) * y_dim, k * y_dim * y_dim : (k + 1) * y_dim * y_dim] = torch.kron(torch.eye(y_dim, device=device), fi[k * y_dim : (k + 1) * y_dim].unsqueeze(1).T)
                for l in range(x_channel):
                    Cxi[k * y_dim : (k + 1) * y_dim, l * x_dim * x_dim : (l + 1) * x_dim * x_dim] = -torch.kron(dfi[k * y_dim : (k + 1) * y_dim, l * x_dim : (l + 1) * x_dim], xi[l * x_dim : (l + 1) * x_dim].unsqueeze(1).T)
            CTC += (Cxi + Cyi).T @ (Cxi + Cyi)
        U, S, V = torch.svd(CTC)
        A = torch.zeros((x_channel * x_dim * x_dim, x_channel * x_dim, x_channel * x_dim), device=device)
        for i in range(x_channel):
            A[:, i * x_dim : (i + 1) * x_dim, i * x_dim : (i + 1) * x_dim] = V[i * x_dim * x_dim : (i + 1) * x_dim * x_dim, :].T.reshape(x_channel * x_dim * x_dim, x_dim, x_dim)
        sigma = torch.sqrt(S)
        # for param in mlp.parameters():
            # param.requires_grad = True
        print("End symmetry discovery.")
        return A.detach().cpu(), sigma.detach().cpu()

def discovery_inertia_tensor(net, discovery_dataloader, device):
    # for param in net.parameters():
        # param.requires_grad = False 
    print("Start symmetry discovery.")
    for x, y in discovery_dataloader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]
        x_channel = x.shape[1]
        x_dim = x.shape[2]
        y_dim_1 = y.shape[1]
        y_dim_2 = y.shape[2]
        x = x.reshape(batch_size, x_channel * x_dim)
        y = y.reshape(batch_size, y_dim_1 * y_dim_2)
        CTC = torch.zeros((x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2, x_dim * x_dim + y_dim_1 * y_dim_1 + y_dim_2 * y_dim_2), device=device)
        for i in trange(batch_size):
            xi = x[i, :]
            xi.requires_grad = True
            fi = net(xi)
            dfi = torch.zeros((y_dim_1 * y_dim_2, x_channel * x_dim), device=device)
            for j in range(y_dim_1 * y_dim_2):
                dfi[j, :] = torch.autograd.grad(fi[j], xi, retain_graph=True)[0]
            Cxi = torch.zeros((y_dim_1 * y_dim_2, x_dim * x_dim), device=device)
            for j in range(x_channel):
                Cxi -= torch.kron(dfi[:, j * x_dim : (j + 1) * x_dim], xi[j * x_dim : (j + 1) * x_dim].unsqueeze(1).T)
            Fi = fi.reshape(y_dim_1, y_dim_2)
            Cy1i = torch.kron(torch.eye(y_dim_1, device=device), Fi.T.contiguous())
            Cy2i = torch.zeros((y_dim_1 * y_dim_2, y_dim_2 * y_dim_2), device=device)
            for j in range(y_dim_1):
                Cy2i[j * y_dim_2 : (j + 1) * y_dim_2, :] = torch.kron(torch.eye(y_dim_2, device=device), Fi[j].unsqueeze(0))
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
        return Ax.detach().cpu(), Ay1.detach().cpu(), Ay2.detach().cpu(), sigma.detach().cpu()

def discovery_inertia_vector(net, discovery_dataloader, device):
    # for param in net.parameters():
        # param.requires_grad = False 
    print("Start symmetry discovery.")
    for x, y in discovery_dataloader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]
        x_channel = x.shape[1]
        x_dim = x.shape[2]
        y_channel = y.shape[1]
        y_dim = y.shape[2]
        x = x.reshape(batch_size, x_channel * x_dim)
        y = y.reshape(batch_size, y_channel * y_dim)
        CTC = torch.zeros((x_dim * x_dim + y_channel * y_dim * y_dim, x_dim * x_dim + y_channel * y_dim * y_dim), device=device)
        for i in trange(batch_size):
            xi = x[i, :]
            xi.requires_grad = True
            fi = net(xi)
            dfi = torch.zeros((y_channel * y_dim, x_channel * x_dim), device=device)
            for j in range(y_channel * y_dim):
                dfi[j, :] = torch.autograd.grad(fi[j], xi, retain_graph=True)[0]
            Cxi = torch.zeros((y_channel * y_dim, x_dim * x_dim), device=device)
            Cyi = torch.zeros((y_channel * y_dim, y_channel * y_dim * y_dim), device=device)
            for k in range(y_channel):
                Cyi[k * y_dim : (k + 1) * y_dim, k * y_dim * y_dim : (k + 1) * y_dim * y_dim] = torch.kron(torch.eye(y_dim, device=device), fi[k * y_dim : (k + 1) * y_dim].unsqueeze(1).T)
                for l in range(x_channel):
                    Cxi[k * y_dim : (k + 1) * y_dim, :] -= torch.kron(dfi[k * y_dim : (k + 1) * y_dim, l * x_dim : (l + 1) * x_dim], xi[l * x_dim : (l + 1) * x_dim].unsqueeze(1).T)
            CTC[: x_dim * x_dim, : x_dim * x_dim] += Cxi.T @ Cxi
            CTC[: x_dim * x_dim, x_dim * x_dim :] += Cxi.T @ Cyi
            CTC[x_dim * x_dim :, : x_dim * x_dim] += Cyi.T @ Cxi
            CTC[x_dim * x_dim :, x_dim * x_dim :] += Cyi.T @ Cyi
        U, S, V = torch.svd(CTC)
        Ax = V[: x_dim * x_dim, :].T.reshape(x_dim * x_dim + y_channel * y_dim * y_dim, x_dim, x_dim)
        Ay = torch.zeros((x_dim * x_dim + y_channel * y_dim * y_dim, y_channel * y_dim, y_channel * y_dim), device=device)
        for i in range(y_channel):
            Ay[:, i * y_dim : (i + 1) * y_dim, i * y_dim : (i + 1) * y_dim] = V[x_dim * x_dim + i * y_dim * y_dim : x_dim * x_dim + (i + 1) * y_dim * y_dim, :].T.reshape(x_dim * x_dim + y_channel * y_dim * y_dim, y_dim, y_dim)
        sigma = torch.sqrt(S)
        print("End symmetry discovery.")
        return Ax.detach().cpu(), Ay.detach().cpu(), sigma.detach().cpu()

def discovery_top_quark_tagging(net, discovery_dataloader, device):
    for param in net.parameters():
        param.requires_grad = False
    print("Start symmetry discovery.")
    for x, y in discovery_dataloader:
        x = x.to(device)
        y = y.to(device)
        batch_size = x.shape[0]
        x_channel = x.shape[1]
        x_dim = x.shape[2]
        y_channel = y.shape[1]
        y_dim = y.shape[2]
        x = x.reshape(batch_size, x_channel * x_dim)
        y = y.reshape(batch_size, y_channel * y_dim)
        CTC = torch.zeros((x_dim * x_dim, x_dim * x_dim), device=device)
        for i in trange(batch_size):
            xi = x[i, :]
            xi.requires_grad = True
            fi = net(xi)
            dfi = torch.zeros((y_channel * y_dim, x_channel * x_dim), device=device)
            for j in range(y_channel * y_dim):
                dfi[j, :] = torch.autograd.grad(fi[j], xi, retain_graph=True)[0]
            Cxi = torch.zeros((y_channel * y_dim, x_dim * x_dim), device=device)
            for k in range(y_channel):
                for l in range(x_channel):
                    Cxi[k * y_dim : (k + 1) * y_dim, :] -= torch.kron(dfi[k * y_dim : (k + 1) * y_dim, l * x_dim : (l + 1) * x_dim], xi[l * x_dim : (l + 1) * x_dim].unsqueeze(1).T)
            CTC += Cxi.T @ Cxi
        U, S, V = torch.svd(CTC)
        A = V.T.reshape(x_dim * x_dim, x_dim, x_dim)
        sigma = torch.sqrt(S)
        for param in net.parameters():
            param.requires_grad = True
        print("End symmetry discovery.")
        return A.detach().cpu(), sigma.detach().cpu()

def discovery_mnist(net, discovery_dataloader, device):
    for param in net.parameters():
        param.requires_grad = False
    print("Start symmetry discovery.")
    for image, label in discovery_dataloader:
        image = image.to(device)
        label = label.to(device)
        batch_size = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]
        x = torch.zeros((height, width, 2), device=device)
        x[:, :, 0], x[:, :, 1] = torch.meshgrid(torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing='ij')
        dIdx = torch.zeros((batch_size, height - 2, width - 2, 2), device=device)
        dIdx[:, :, :, 0] = (image[:, 0, 2:, 1:-1] - image[:, 0, :-2, 1:-1]) / (x[2:, 1:-1, 0] - x[:-2, 1:-1, 0])
        dIdx[:, :, :, 1] = (image[:, 0, 1:-1, 2:] - image[:, 0, 1:-1, :-2]) / (x[1:-1, 2:, 1] - x[1:-1, :-2, 1])
        x = x[1:-1, 1:-1, :]
        image.requires_grad = True
        image.retain_grad()
        F = net(image)
        F.backward(torch.ones_like(F))
        dFdI = image.grad[:, 0, 1:-1, 1:-1]
        dFdx = torch.einsum('bhw,bhwi->bhwi', dFdI, dIdx)
        C = torch.einsum('bhwi,hwj->bij', dFdx, x).reshape(batch_size, -1)
        U, S, V = torch.svd(C)
        A = V.T.reshape(4, 2, 2)
        for param in net.parameters():
            param.requires_grad = True
        print("End symmetry discovery.")
        return A.detach().cpu(), S.detach().cpu()
