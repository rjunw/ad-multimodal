import torch
import torch.nn.functional as F
import tqdm 
import numpy as np

def epoch_contrastive(model, criterion, loader, epoch, 
                       w = 1, optimizer = None, device = 'cpu'):
    """
    standard contrastive epoch
    """
    # print(criterion)
    if optimizer:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Val'

    train_loss = []
    batches = tqdm(enumerate(loader), total=len(loader))
    batches.set_description("Epoch NA: Loss (NA)")

    for batch_idx, (im, modes, y) in batches:
        im, modes, y = im.to(device), [mode.to(device) for mode in modes], y.to(device)
        z_im, z_modes = model(im, modes)
        loss = 0
        for z_mode in z_modes:
            loss = loss + w * criterion(z_im, z_mode)#, y.to(torch.int64))
            
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(loss.item())
        batches.set_description(
            "Epoch {:d}: {:s} Loss ({:.2e})".format(
                epoch, mode, loss.item()
            )
        )

    return np.mean(train_loss)

def epoch_standard(model, criterion, loader, epoch, optimizer = None, device = 'cpu'):
    """
    standard epoch
    """
    if optimizer:
        model.train()
        mode = 'Train'
    else:
        model.eval()
        mode = 'Val'

    train_loss = []
    batches = tqdm(enumerate(loader), total=len(loader))
    batches.set_description("Epoch NA: Loss (NA) ACC (NA)")

    count = 0
    correct = 0

    for batch_idx, (x, _, y) in batches:
        x, y = x.to(device), y.to(device)
        z = model(x)
        # print(z.shape, y.shape, type(z), type(y))
        loss = F.cross_entropy(z, y.to(torch.int64)) # criterion isnt working??
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        correct += (z.max(axis = 1).indices == y).float().sum()
        count += y.shape[0]

        train_loss.append(loss.item())
        batches.set_description(
            "Epoch {:d}: {:s} Loss ({:.2e}) ACC ({:.2e})".format(
                epoch, mode, loss.item(), 100 * correct / count
            )
        )

    return np.mean(train_loss), (100 * correct/count).detach().cpu().numpy()
