from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

from utilities.evaluation import calculate_psnr, calculate_ssim, SSIM_rgb




def mkdict(state_dict, validation_losses, epoch, optimizer, scheduler=None):
    out = {
        'state_dict': state_dict,
        'validation_losses': validation_losses,
        'epoch': epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    return out




def plot_loss(break_point, Valid_Loss_list, learning_rate_list, Loss_list):
    plt.figure(dpi=500)

    plt.subplot(211)
    x = range(break_point)
    y = Loss_list
    plt.plot(x, y, 'ro-', label='Train Loss')
    plt.plot(range(break_point), Valid_Loss_list, 'bs-', label='Valid Loss')
    plt.ylabel('Loss')
    plt.xlabel('epochs')

    plt.subplot(212)
    plt.plot(x, learning_rate_list, 'ro-', label='Learning rate')
    plt.ylabel('Learning rate')
    plt.xlabel('epochs')

    plt.legend()
    plt.show()




class Mixed_Loss:
    def __init__(self, alpha=.2, beta=.9, epsilon=1e-10):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.ssim = SSIM_rgb()

    def calculate(self, ground_truths, outputs):
        return self.alpha * F.mse_loss(ground_truths, outputs) + self.beta * (1 - self.ssim(ground_truths, outputs))




def train_model_progressive(model_list, 
                            train_loaders, 
                            valid_loaders, 
                            device, 
                            optimizer=None, 
                            scheduler=None, 
                            epoch=10,
                            epoch_max = 100,
                            epoch_s=0, 
                            lr=5e-4, 
                            patience=40, 
                            lr_min=4e-6,
                            im_sizes=128):
                
    model_name = model_list[1]
    model = model_list[0]
    stale = 0
    break_point = 0
    if not optimizer: optimizer = optim.RAdam(model.parameters(), lr=lr)
    if not scheduler: scheduler = CosineAnnealingLR(optimizer, T_max=epoch_max, eta_min=lr_min)

    criterion = SSIM_rgb()

    Loss_list = []
    Valid_Loss_list = []
    learning_rate_list = []

    best_valid_loss = 100000
    imsize = im_sizes
    train_loader = train_loaders
    valid_loader = valid_loaders

    for i in range(epoch_s, epoch_max):
    # ---------------Train----------------
        model.train()
        train_losses = []
        print('------------------------------------------')
        print(f'Epoch {i + 1:03d}/{epoch_max:03d}')
        print('Training:')
        for _, batch in enumerate(tqdm(train_loader)):
            inputs, labels = batch
            outputs = model(inputs.to(device))
            loss = criterion(labels.to(device), outputs)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            train_losses.append(loss.item())
            
        train_loss = sum(train_losses) / len(train_losses)
        Loss_list.append(train_loss)
        print(f"[ Train | {i + 1:03d}/{epoch_max:03d} ] Mixed Loss = {train_loss:.5f}")
        
        scheduler.step()
        for param_group in optimizer.param_groups:
            learning_rate_list.append(param_group["lr"])
            print('                    Learning Rate %f' % param_group["lr"])

        print('')
        
    # -------------Validation-------------
        print('Validation:')
        model.eval()
        valid_losses = []
        psnrs = []
        ssims = []
        for batch in tqdm(valid_loader):
            inputs, labels = batch

            with torch.no_grad():
                outputs = model(inputs.to(device))
            loss = criterion(labels.to(device), outputs)
            psnr = calculate_psnr(labels, outputs.cpu().detach())
            ssim = calculate_ssim(labels, outputs.cpu().detach())

            valid_losses.append(loss.item())
            psnrs.append(psnr)
            ssims.append(ssim)
        
        valid_loss = sum(valid_losses) / len(valid_losses)
        psnr_mean = sum(psnrs) / len(psnrs)
        ssim_mean = sum(ssims) / len(ssims)

        Valid_Loss_list.append(valid_loss)
        
        break_point = i + 1
        print(f"[ Valid | {i + 1:03d}/{epoch_max:03d} ] PSNR = {psnr_mean:.5f}")
        print(f"                    SSIM = {ssim_mean:.5f}")

        if valid_loss < best_valid_loss:
            print(f"                    Mixed Loss = {valid_loss:.5f} -> best")
            print(f'Best model found at epoch {i+1}, saving model')

            net = mkdict(model.state_dict(), Valid_Loss_list, epoch, optimizer, scheduler)
            torch.save(net, f'trained_models/{model_name}/best_model_{imsize}p_{i+1}e.pth')

            best_valid_loss = valid_loss
            stale = 0
        else:
            print(f"                    Mixed Loss = {valid_loss:.5f}")
            stale += 1
            if stale > patience:
                print(f'No improvement {patience} consecutive epochs, early stopping.')
                break

        net = mkdict(model.state_dict(), Valid_Loss_list, epoch, optimizer, scheduler)
        torch.save(net, f'trained_models/{model_name}/best_model_{imsize}p_{i+1}e.pth')

    result = {
        'break_point':break_point,
        'Valid_Loss_list':Valid_Loss_list,
        'learning_rate_list':learning_rate_list,
        'Loss_list':Loss_list,
    }

    return result