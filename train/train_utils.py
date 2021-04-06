import torch

def data_to_device(data, device):
    """Recursively transfers all tensors in dictionary to given device"""
    if isinstance(data, torch.Tensor):
        res = data.to(device)
    elif isinstance(data, dict):
        res = {k: data_to_device(v, device) for k, v in data.items()}
    else:
        res = data
    return res


if __name__ == '__main__':
    x = {'test_key': torch.tensor([1, 3, 4, 5])}
    print(data_to_device(x, device='cuda:0'))