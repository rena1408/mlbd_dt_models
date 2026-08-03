import time
import os

import numpy as np
from sklearn import metrics as metrics


import torch

import pandas as pd


# ## Checking available cpus/gpus
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(torch.cuda.is_available())
print(f'Using device: {device}')


if device == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

def calc_evals(truth_file, test_score_file, out_path):

    test_flags = np.load(truth_file)

    test_scores = np.load(test_score_file)

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


    eval_metrics_array = [[accuracy, recall, precision, specificity, youden_j_statistic, f1_score, auc_score,
                                    fp, fn, tn, tp]]
    
        
    eval_metrics_df = pd.DataFrame(eval_metrics_array, columns = ["accuracy", "recall", "precision", "specificity", "youden_j_statistic", "f1_score", "auc_score",
                                     "fp", "fn", "tn", "tp"])

    eval_metrics_df.to_csv(f"{out_path}/GNN_bootstrap_evaluation_metrics.csv")

    end = time.time()

    accuracy_list = eval_metrics_df["accuracy"]
    auc_list = eval_metrics_df["auc_score"]

    #print(f"The accuracies are {accuracy_list}. \n")
    print(f"The test accuracy is {np.mean(accuracy_list)}. \n")

    print(f"The test AUC is {np.mean(auc_list)}. \n")


if __name__ == '__main__':
    calc_evals(test_score_file= "rn325/my_analysis/mlbd_dt_models/model_out_data/GNN_out/test_scores.npy", truth_file=
               "rn325/my_analysis/mlbd_dt_models/model_out_data/GNN_out/truth.npy", out_path="rn325/my_analysis/mlbd_dt_models/comparison_study/eval_metrics")