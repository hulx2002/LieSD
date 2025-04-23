import torch

def choose_nonlinearity(name):
    nl = None
    if name == 'tanh':
        nl = torch.tanh
    elif name == 'relu':
        nl = torch.relu
    elif name == 'sigmoid':
        nl = torch.sigmoid
    elif name == 'softplus':
        nl = torch.nn.functional.softplus
    elif name == 'selu':
        nl = torch.nn.functional.selu
    elif name == 'elu':
        nl = torch.nn.functional.elu
    elif name == 'swish':
        nl = lambda x: x * torch.sigmoid(x)
    else:
        raise ValueError("nonlinearity not recognized")
    return nl

def rotate_90(x):
    rotation_matrix = torch.tensor([[0., -1.], [1., 0.]])
    x_rotate = torch.einsum('ij,bcj->bci', rotation_matrix, x.reshape(x.shape[0], -1, 2)).reshape(x.shape)
    return x_rotate