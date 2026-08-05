import time
import os

import numpy as np
from sklearn import metrics as metrics


import torch
import torchinfo
from torch_geometric.loader import DataLoader

from utils.classes import GNNClassifier
from utils.general_functions import load_data

import pandas as pd


# ## Checking available cpus/gpus
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(torch.cuda.is_available())
print(f'Using device: {device}')


if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

def model_eval(name, conv_layer_dims = [16, 32, 64, 128], bootstrap = True, n_bootstrap = 10):

    test_sample = load_data("test")

    start = time.time()

    N_dims = 3
    input_dim = N_dims  # x, y, q

    PATH = f"DM_GNN/model_states/{name}"

    out_path = f"DM_GNN/outputs/eval/{name}"
    os.makedirs(out_path, exist_ok = True)

    trials_out_path = f"DM_GNN/model_states/outputs/eval/{name}/separate"
    os.makedirs(trials_out_path, exist_ok = True)

    model = GNNClassifier(input_dim=input_dim, hidden_dims=conv_layer_dims, output_dim=1).to(device)
    state_dict = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    torchinfo.summary(model)

    if bootstrap == True:
        n_boot = n_bootstrap
    else:
        n_boot = 1

    eval_metrics_array = np.zeros((n_boot, 11))

    for trial in range(n_boot):
    
        if bootstrap == True and trial > 0:
            rng = np.random.default_rng()
            chosen = rng.choice(len(test_sample), len(test_sample))

            # create a bootstrapping sample
            new_data = [test_sample[i] for i in chosen]

        else:
            new_data = test_sample

        batch_size = 32
        test_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True)

        test_flags, test_scores = [], []

        for batch in test_loader:
            model.eval()
            with torch.no_grad():
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)
                scores = torch.sigmoid(outputs).squeeze()
                test_scores.extend(scores.cpu().numpy())
                test_flags.extend(batch.y.cpu().numpy())

        np.save(trials_out_path + f'/test_scores_{trial}.npy', np.array(test_scores))
        np.save(trials_out_path + f'/truth_{trial}.npy', np.array(test_flags))


        # performance metrics:

        roc_curve = metrics.roc_curve(test_flags, test_scores)
        auc_score = metrics.auc(roc_curve[0], roc_curve[1])


        predicted_class = (np.array(test_scores) >= 0.5).astype(int)
        confusion_matrix_arrs = metrics.confusion_matrix(test_flags, predicted_class)

        fp = confusion_matrix_arrs[0, 1]
        fn = confusion_matrix_arrs[1, 0]
        tn = confusion_matrix_arrs[0, 0]
        tp = confusion_matrix_arrs[1, 1]

        accuracy = (tp+tn)/(tp+fp+tn+fn)
        recall = tp / (tp + fn)         # recall = sensitivity = true positive rate
        precision = tp / (tp + fp)
        specificity = tn/(tn + fp)

        youden_j_statistic = recall + specificity -1
        f1_score = 2*(precision*recall)/(precision+recall)


        eval_metrics_array[trial] = [accuracy, recall, precision, specificity, youden_j_statistic, f1_score, auc_score,
                                     fp, fn, tn, tp]
        
        
    eval_metrics_df = pd.DataFrame(eval_metrics_array, columns = ["accuracy", "recall", "precision", "specificity", "youden_j_statistic", "f1_score", "auc_score",
                                     "fp", "fn", "tn", "tp"])

    eval_metrics_df.to_csv(f"{out_path}/bootstrap_evaluation_metrics")

    end = time.time()

    print(f"Ran {n_boot} trials in {end - start} seconds. \n")

    accuracy_list = eval_metrics_df["accuracy"]

    print(f"The accuracies are {accuracy_list}. \n")
    print(f"Number of bootstrapping samples: {n_boot}")
    print(f"The mean accuracy is {np.mean(accuracy_list)}. \n")
    print(f"With a variance of {np.var(accuracy_list)}.")

    print(eval_metrics_df.head())

if __name__ == '__main__':
    model_eval("model_gnn_547496", n_bootstrap = 10)