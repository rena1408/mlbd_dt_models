import time
import sys
import os

var = os.environ["SINGULARITY_CACHEDIR"]

cluster = ''.join(filter(str.isdigit, var))

print(cluster)
parent_d = "/vols/sbn/uboone/rn325/my_analysis/mlbd_dt_models/DM_GNN/outputs"
path = os.path.join(parent_d, cluster)
os.mkdir(path)

print("Currently active Python virtual environment:", sys.prefix)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics as metrics


import torch
import torchinfo
import torch.nn as nn

from torch_geometric.loader import DataLoader

from utils import classes
from utils import general_functions

# ## Checking available cpus/gpus
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(torch.cuda.is_available())
print(f'Using device: {device}')


if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

def train_function(nepochs, scheduler, gamma, optimizer, model, train_loader,
                   criterion, val_loader, early_stopper):

    train_losses, val_losses, train_accuracies, val_accuracies, model_states, truth_labels = [], [], [], [], [], []
    
    for epoch in range(nepochs):
            print(f'Epoch {epoch+1}/{nepochs}')
            if scheduler is not None and gamma != 1:
                print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
            printProgressBar(0, len(train_loader), prefix = 'Progress:', suffix = 'Complete', length = 50)
    
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            epoch_scores = []
            j = 1
    
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
    
                # Note here that we are passing skip_output_activation=True to the model's forward function.
                # This is because we are using BCEWithLogitsLoss as our loss function, which expects
                # raw logits rather than a score between 0 and 1. Using logits in loss leads to more numerically stable training.
                outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)#, device = device)
                scores = torch.sigmoid(outputs).squeeze()
                loss = criterion(outputs.squeeze(), batch.y.float())
                loss.backward()
                optimizer.step()
    
                epoch_scores.append(scores)
    
                loss_value = loss.mean().item()
                epoch_loss += loss_value
                acc = ((scores > 0.5) == batch.y).float().mean().item()
                epoch_acc += acc
    
                if epoch == 0:
                    truth_labels.extend(batch.y.cpu())
                
                if j/len(train_loader) in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    printProgressBar(j, len(train_loader), prefix = 'training progress:', suffix = 'Complete', length = 50)
                j += 1
            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader)
    
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            model_states.append(model.state_dict())
    
            # Validation
            model.eval()
            epoch_val_loss = 0
            epoch_val_acc = 0
            for batch in val_loader:
                with torch.no_grad():
                    batch = batch.to(device)
                    outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)
                    scores = torch.sigmoid(outputs).squeeze()
                    loss = criterion(outputs.squeeze(), batch.y.float())
                    preds = (scores > 0.5).long()
                    acc = (preds == batch.y).float().mean().item()
                    epoch_val_acc += acc
                    epoch_val_loss += loss.mean().item()
            epoch_val_loss /= len(val_loader)
            epoch_val_acc /= len(val_loader)
    
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
    
            scheduler.step(epoch_val_loss) if scheduler is not None and gamma != 1 else None
            print(f"Epoch {epoch+1}/{nepochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
    
            if early_stopper.early_stop(epoch_val_loss):
                print(f"Stopped training due to early stopping criteria. \n Stopped at epoch {epoch+1}.")
                break

    return model, truth_labels, train_losses, train_accuracies, model_states, val_losses, val_accuracies    

def train_GNN(data_path = "", train_epochs = 50, frac_train = 0.85, 
              conv_layer_dims = [16, 32, 64, 128]):

    if data_path == "":
        all_graphs = load_data("train")
        #test_sample = load_data("test")
    else:
        all_graphs = load_data("train", path = data_path)
        #test_sample = load_data("test", path = data_path)

    np.random.seed(42)
    np.random.shuffle(all_graphs)

    start = time.time()

    print("Splitting data into train, test, and validation sets...")

    train_sample, val_sample = train_test_split(all_graphs, test_size=1-frac_train, random_state=42)# , shuffle = True) #random_state=32, 
    
    print(f"Number of training graphs: {len(train_sample)}")
    print(f"Number of validation graphs: {len(val_sample)}")
    #print(f"Number of testing graphs: {len(test_sample)}")

    input_dim = 3  # x, y, q

    
    model = GNNClassifier(input_dim=input_dim, hidden_dims=conv_layer_dims, output_dim=1).to(device)

    dummy_data = torch.randn((10, input_dim)).to(device)  # 10 nodes, input_dim features
    dummy_edges = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]], dtype=torch.long).to(device)  # example edge index
    dummy_batch = torch.zeros(10, dtype=torch.long).to(device)  # all nodes belong to the same graph
    torchinfo.summary(model, input_data=(dummy_data, dummy_edges, dummy_batch))


    batch_size = 32      # implement early stopping criteria

    lr = 1e-4
    gamma = 0.1 # learning rate decay factor, set to 1 to disable decay

    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 5)
    early_stopper = EarlyStopper(patience=5, min_delta=0)   #he's using patience = 5

    train_loader = DataLoader(train_sample, batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(test_sample, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sample, batch_size=batch_size, shuffle=True)

    model, truth_labels, train_losses, train_accuracies, model_states, val_losses, val_accuracies = train_function(train_epochs, scheduler, gamma, optimizer, model, train_loader, 
                   criterion, val_loader, early_stopper)

    plt.hist(truth_labels)
    plt.xlabel('label')
    plt.ylabel('count')
    plt.title('count per class in training')
    plt.grid()
    plt.show()

    title = path + "/training_truth_labels.jpg"
    plt.savefig(title)
    plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.show()

    title = path + "/train_val_loss.jpg"
    plt.savefig(title)
    plt.close()

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    title = path + "/train_val_acc.jpg"
    plt.savefig(title)
    plt.close()

    optimal_epoch = np.argmin(val_losses) # Decide how to choose the optimal epoch - np.argmin(val_losses) or np.argmax(val_accuracies)
    print(f"Best model found at epoch {optimal_epoch+1} with validation loss {val_losses[optimal_epoch]:.4f} and validation accuracy {val_accuracies[optimal_epoch]:.4f}")

    best_model_state = model_states[optimal_epoch]

    model.load_state_dict(best_model_state)

    PATH = f"/vols/sbn/uboone/rn325/my_analysis/mlbd_dt_models/DM_GNN/model_states/{cluster}"

    print(f"Saving the best model at {PATH}...")
    torch.save(model.state_dict(), PATH)

    end = time.time()

    print(f"The model trained for {end - start} seconds over {train_epochs} epochs.")

    return cluster

if __name__ == '__main__':
    train_GNN()