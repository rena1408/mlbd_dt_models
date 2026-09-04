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
import scipy.stats
from MLstatkit.stats import Delong_test

train_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def written_Delong_test(true, prob_A, prob_B):
    """
    TAKEN FROM https://medium.com/statistics-in-machine-learning/comparing-roc-curves-in-machine-learning-model-with-delongs-test-a-practical-guide-using-python-e70b5d20abde
    Perform DeLong's test for comparing the AUCs of two models.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        True binary labels in range {0, 1}.
    prob_A : array-like of shape (n_samples,)
        Predicted probabilities by the first model.
    prob_B : array-like of shape (n_samples,)
        Predicted probabilities by the second model.

    Returns
    -------
    z_score : float
        The z score from comparing the AUCs of two models.
    p_value : float
        The p value from comparing the AUCs of two models.

    Example
    -------
    >>> true = [0, 1, 0, 1]
    >>> prob_A = [0.1, 0.4, 0.35, 0.8]
    >>> prob_B = [0.2, 0.3, 0.4, 0.7]
    >>> z_score, p_value = Delong_test(true, prob_A, prob_B)
    >>> print(f"Z-Score: {z_score}, P-Value: {p_value}")
    """

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        T2[J] = T + 1
        return T2

    def compute_ground_truth_statistics(true):
        assert np.array_equal(np.unique(true), [0, 1]), "Ground truth must be binary."
        order = (-true).argsort()
        label_1_count = int(true.sum())
        return order, label_1_count

    # Prepare data
    order, label_1_count = compute_ground_truth_statistics(np.array(true))
    sorted_probs = np.vstack((np.array(prob_A), np.array(prob_B)))[:, order]

    # Fast DeLong computation starts here
    m = label_1_count  # Number of positive samples
    n = sorted_probs.shape[1] - m  # Number of negative samples
    k = sorted_probs.shape[0]  # Number of models (2)

    # Initialize arrays for midrank computations
    tx, ty, tz = [np.empty([k, size], dtype=np.float64) for size in [m, n, m + n]]
    for r in range(k):
        positive_examples = sorted_probs[r, :m]
        negative_examples = sorted_probs[r, m:]
        tx[r, :], ty[r, :], tz[r, :] = [
            compute_midrank(examples) for examples in [positive_examples, negative_examples, sorted_probs[r, :]]
        ]

    # Calculate AUCs
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    # Compute variance components
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    # Compute covariance matrices
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    # Calculating z-score and p-value
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, delongcov), l.T)).flatten()
    p_value = scipy.stats.norm.sf(abs(z)) * 2

    z_score = -z[0].item()
    p_value = p_value[0].item()

    return z_score, p_value


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

def load_truth_test_data(type, truth, test, l_signal = 2686, l_bkg = 4121):

    t_1 = np.load(truth).flatten()
    t_2 = np.load(test).flatten()

    ts = np.transpose([t_1, t_2])

    cl1_df = pd.DataFrame(ts, columns = ["truth", "test"])

    ## pop events listed in exclude thingy?
    cl1_bkg = cl1_df.loc[cl1_df['truth'] == 0][:l_bkg]          # NOTE THAT [:l_bkg] IS A TEMPORARY FIX AND WILL PRODUCE INACCURATE RESULTS
    cl1_signal = cl1_df.loc[cl1_df['truth'] == 1][:l_signal]

    print(f" The dataset contains {len(cl1_bkg)} background events.")
    print(f" The dataset contains {len(cl1_signal)} signal events.")

    cl1_ordered_df = pd.concat([cl1_bkg, cl1_signal], ignore_index=True)

    return cl1_ordered_df

models = ["CNN", "GT"]


cl1_df = load_truth_test_data("GT", "rn325/my_analysis/mlbd_dt_models/this_model/GT/separate/truth_0.npy", "rn325/my_analysis/mlbd_dt_models/this_model/GT/separate/test_scores_0.npy")
cl2_df = load_truth_test_data("GNN", "rn325/my_analysis/mlbd_dt_models/model_out_data/GNN_out/truth.npy", "rn325/my_analysis/mlbd_dt_models/model_out_data/GNN_out/test_scores.npy")

print(np.all(cl1_df["truth"].to_numpy() == cl2_df["truth"].to_numpy()))

truth = cl1_df["truth"].to_numpy()

cl1_scores = cl1_df["test"].to_numpy()
cl2_scores = cl2_df["test"].to_numpy()


cl1 = (cl1_scores > 0.5)
cl2 = (cl2_scores > 0.5)


data = np.array([truth, cl1, cl2])


m = mcnemar_data(data, "sensitivity")
n = mcnemar_data(data, "specificity")

z_ml, p_ml, _, _, _, _, _ = Delong_test(truth, cl1_scores, cl2_scores)
print(f"DeLong, AUC stuff, z = {z_ml}, p = {p_ml}")

z, p = written_Delong_test(truth, cl1_scores, cl2_scores)

print(f"DeLong, AUC stuff, z = {z}, p = {p}")

print(f"As p<0.05 is {p<0.05}, the difference in evaluation metrics is ")
if p<0.05:
    print("significant.")
else:
    print("insignificant.")

# confusion_matrix_arrs = [[237,1965], [1990, 174]]

# tp = confusion_matrix_arrs[0][1]
# tn = confusion_matrix_arrs[1][0]
# fn = confusion_matrix_arrs[0][0]
# fp = confusion_matrix_arrs[1][1]

# accuracy = (tp+tn)/(tp+fp+tn+fn)
# # recall = metrics.recall_score()
# # f1_score = metrics.f1_score()
# # accuracy = metrics.accuracy_score()
# # precision = metrics.precision_score()
# recall = tp / (tp + fn) #recall = sensitivity = true positive rate
# precision = tp / (tp + fp)
# specificity = tn/(tn + fp) #specificity = true negative rate

# youden_j_statistic = recall + specificity -1
# f1_score = 2*(precision*recall)/(precision+recall)
# # kappa = #omg this formula is so long
# # MCC = #maybe?

# #calculate AUC


# # roc_curve = metrics.roc_curve(test_flags, test_scores)
# # auc_score = metrics.auc(roc_curve[0], roc_curve[1])

# print(f"recall: {recall}")
# print(f"precision: {precision}")
# print(f"accuracy: {accuracy}")
# print(f"specificity: {specificity}")
# print(f"youden_j_statistic: {youden_j_statistic}")
# print(f"f1_score: {f1_score}")