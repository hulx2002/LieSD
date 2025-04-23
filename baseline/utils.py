import torch

# scale the tensor to have dummy position equal to 1
def affine_coord(tensor, dummy_pos=None):
    # tensor: B*T*K
    if dummy_pos is not None:
        return tensor / tensor[..., dummy_pos].unsqueeze(-1)
    else:
        return tensor

def rotate_90(x):
    rotation_matrix = torch.tensor([[0., -1.], [1., 0.]])
    x_rotate = torch.einsum('ij,bcj->bci', rotation_matrix, x.reshape(x.shape[0], -1, 2)).reshape(x.shape)
    return x_rotate