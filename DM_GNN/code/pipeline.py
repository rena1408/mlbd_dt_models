from train_GNN import train_GNN
from eval_GNN import model_eval

if __name__ == '__main__':
    model_name = train_GNN()
    model_eval(name = model_name)
