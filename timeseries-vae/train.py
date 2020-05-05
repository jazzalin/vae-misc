import numpy as np
import torch
from torch import nn, optim
from torch import distributions
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import os
from model import Vrae
from data.dataloader import ExerciseDataset


def loss_function(x_decoded, x, mean, logvar, criterion):
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    recon_loss = criterion(x_decoded, x)
    return kl_loss + recon_loss, recon_loss, kl_loss


def training(train_loader, net, optimizer, criterion, writer, epoch):
    
    epoch_loss = 0
    net.train()
    
    len_train = len(train_loader)
    
    for i, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        x_batch, y_batch = batch
        x_batch = x_batch.permute(1, 0, 2)
        x_batch = x_batch.float().cuda()
        y_batch = y_batch.long().cuda()
        
        out, mu, logvar = model(x_batch)
#         print(out.shape, x_batch.shape, mu.shape, logvar.shape)
        
        total_loss, rec_loss, kl_loss = loss_function(out, x_batch, mu, logvar, criterion)
        
        # Loss
        total_loss.backward()
        optimizer.step()
        
        
        
        epoch_loss += total_loss.item()
        iteration = epoch * len_train + i
        writer.add_scalars('train/loss', {'total_loss': total_loss.item(), "kl_loss": kl_loss.item(), "rec_loss": rec_loss.item()}, iteration)
    
    return epoch_loss / len(train_loader)


def evaluate(val_loader, net, criterion, writer, epoch):
    
    epoch_loss = 0
    net.eval()

    len_val = len(val_loader)
    
    with torch.no_grad():
        for i, val in enumerate(val_loader):
            x_val, y_val = val
            x_val = x_val.permute(1, 0, 2)
            x_val = x_val.float().cuda()
            y_val = y_val.long().cuda()
    
            out, mu, logvar = model(x_val)
#             print(out.shape, mu.shape, logvar.shape)

            total_loss, rec_loss, kl_loss = loss_function(out, x_batch, mu, logvar, criterion)
            
            # Loss
            loss_graph.append(total_loss.item())
            rec_graph.append(rec_loss.item())
            kl_graph.append(kl_loss.item())
              
            epoch_loss += total_loss.item()
            # iteration = epoch * len_val + i
            # writer.add_scalar('data/loss', {'val_loss': total_loss.item()}, iteration)

    return epoch_loss / len(val_loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: read configs from config.yaml
    N_EPOCHS = 40
    WORKERS = 6
    hidden_size = 90
    hidden_layer_depth = 1
    latent_length = 20
    batch_size = 32
    learning_rate = 0.0005
    # n_epochs = 150
    dropout_rate = 0.2
    num_features = 3
    output_size = 3 # would be 9 for classifier
    sequence_length = 200
    writer = SummaryWriter()

    # Dataloading
    dataset = ExerciseDataset("./data/cropped_resampled_acc_nar.npy", length=200)
    train_loader = DataLoader(dataset, batch_size=32, num_workers=WORKERS, shuffle=True, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=4, num_workers=WORKERS, shuffle=False, drop_last=True)

    # Model
    model = Vrae(num_features, hidden_size, hidden_layer_depth, latent_length, sequence_length, output_size, batch_size, dropout_rate)
    model = model.cuda()
    criterion = nn.MSELoss()
    criterion.size_average = False
    criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')

    for epoch in tqdm(range(N_EPOCHS)):
    
        train_loss = training(train_loader, model, optimizer, criterion, writer, epoch)
        # writer.add_scalar('train/loss', train_loss, epoch)
        # val_loss = evaluate(test_loader, model, criterion, writer, epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best.pth')
            print(f'Epoch {epoch}: {train_loss:2.2%}')