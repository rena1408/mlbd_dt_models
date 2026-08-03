import torch
import torch.nn as nn

from torch_geometric.nn import conv, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import BatchNorm

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GNNClassifier, self).__init__()
        # input shape (batch, num_nodes, input_dim)

        self.init_batch_norm = BatchNorm(input_dim)

        conv_layers = []
        if isinstance(hidden_dims, int):
            gnn_conv = conv.GraphConv(input_dim, hidden_dims)
            batch_norm = BatchNorm(hidden_dims)
            activation = nn.ReLU()
            conv_layers.append((gnn_conv, batch_norm, activation))
            input_dim = hidden_dims
        else:
            for hidden_dim in hidden_dims:
                gnn_conv = conv.GraphConv(input_dim, hidden_dim)
                batch_norm = BatchNorm(hidden_dim)
                activation = nn.ReLU()
                conv_layers.append((gnn_conv, batch_norm, activation))
                input_dim = hidden_dim
        self.conv_layers = nn.ModuleList([nn.ModuleList(layer) for layer in conv_layers])
        

    
        self.output_layer = nn.Linear(input_dim, output_dim) # do input_dim*2 because of double pooling
        self.output_activation = nn.Sigmoid() if output_dim == 1 else nn.Softmax(dim=1)

    def forward(self, data, edges, batch_indices, skip_output_activation=False):
        # We include an option to skip the output activation function because some loss functions (eg BCEWithLogitsLoss) expect raw logits as input.
        x = self.init_batch_norm(data)

        for gnn_conv, batch_norm, activation in self.conv_layers:
            x = gnn_conv(x, edges)
            x = batch_norm(x)
            x = activation(x)

        #x = torch.cat([global_mean_pool(x, batch_indices), global_max_pool(x, batch_indices)], dim=1)
        x = global_mean_pool(x, batch_indices) 
        x = self.output_layer(x)

        if not skip_output_activation:
            x = self.output_activation(x)
        return x

class EarlyStopper:
    
    '''
    Source - https://stackoverflow.com/a/73704579
    Posted by isle_of_gods, modified by community. See post 'Timeline' for change history
    Retrieved 2026-07-02, License - CC BY-SA 4.0
    '''

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
