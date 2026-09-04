from train_GNN import train_GNN
from eval_GNN import model_eval

if __name__ == '__main__':
    hidden_dims = [32, 64, 128, 256]
    model_name = train_GNN(conv_layer_dims=hidden_dims)
    model_eval(name = model_name, n_bootstrap=1000, conv_layer_dims= hidden_dims)
