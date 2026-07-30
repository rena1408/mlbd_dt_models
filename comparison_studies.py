import time

start = time.time()

import sys
import os

import time
import torch

start = time.time()

import sys
import os

# var = os.environ["SINGULARITY_CACHEDIR"]

# cluster = ''.join(filter(str.isdigit, var))

# print(cluster)
# parent_d = "rn325/GNN/outputs"
# path = os.path.join(parent_d, cluster)
# os.mkdir(path)

# print("Currently active Python virtual environment:", sys.prefix)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as metrics
from statsmodels.stats.contingency_tables import Table2x2 as Table
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

truth = np.random.choice([0, 1], size=(600))
cl1 = np.random.choice([0, 1], size=(600))
cl2 = np.random.choice([0, 1], size=(600))

truth = np.load("/vols/sbn/uboone/rn325/GNN/outputs/eval/model_gnn_232550_0/truth.npy")
cl1_scores = np.load("/vols/sbn/uboone/rn325/GNN/outputs/model_gnn_232550_0/test_flags.npy")
cl2_scores = np.load("/vols/sbn/uboone/rn325/GNN/outputs/model_gnn_3659480/test_flags.npy")

cl1 = (cl1_scores > 0.5)
cl2 = (cl2_scores > 0.5)


data = np.array([truth, cl1, cl2])

def mcnemar_data(data, type):
    print(f"Conducting a McNemar study for {type}...")
    data_trans = data.T
    b = c = 0

    if type == "sensitivity":
        b_comp =[1, 0, 1]
        c_comp = [1, 1, 0]
    elif type == "specificity":
        b_comp =[0, 1, 0]
        c_comp = [0, 0, 1]
    
    for row in data_trans:
        if np.all(row == b_comp):
            #print(row)
            b += 1
        elif np.all(row == c_comp):
            #print(row)
            c += 1
        else:
            continue

    mk_table = [[0, b], [c, 0]]

    print(f"The contingency table: \n {mk_table}")

    if b+c >= 20:
        exact_value = False
        print(f"As b+c is {b+c} >= 20, an approximation is used, i.e. the Chi-squared statistic.")
    else:
        exact_value = True

    mcnem = mcnemar(mk_table, exact = exact_value, correction = True)

    print(f"The results of the McNemar study are as follows: \n {mcnem}")
    print(f"\n")
    return mcnem

m = mcnemar_data(data, "sensitivity")
n = mcnemar_data(data, "specificity")

confusion_matrix_arrs = [[237,1965], [1990, 174]]

tp = confusion_matrix_arrs[0][1]
tn = confusion_matrix_arrs[1][0]
fn = confusion_matrix_arrs[0][0]
fp = confusion_matrix_arrs[1][1]

accuracy = (tp+tn)/(tp+fp+tn+fn)
# recall = metrics.recall_score()
# f1_score = metrics.f1_score()
# accuracy = metrics.accuracy_score()
# precision = metrics.precision_score()
recall = tp / (tp + fn) #recall = sensitivity = true positive rate
precision = tp / (tp + fp)
specificity = tn/(tn + fp) #specificity = true negative rate

youden_j_statistic = recall + specificity -1
f1_score = 2*(precision*recall)/(precision+recall)
# kappa = #omg this formula is so long
# MCC = #maybe?

#calculate AUC


# roc_curve = metrics.roc_curve(test_flags, test_scores)
# auc_score = metrics.auc(roc_curve[0], roc_curve[1])

print(f"recall: {recall}")
print(f"precision: {precision}")
print(f"accuracy: {accuracy}")
print(f"specificity: {specificity}")
print(f"youden_j_statistic: {youden_j_statistic}")
print(f"f1_score: {f1_score}")