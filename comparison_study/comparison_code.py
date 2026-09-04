import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


GNN_df = pd.read_csv("rn325/my_analysis/mlbd_dt_models/comparison_study/eval_metrics/GNN_bootstrap_evaluation_metrics.csv")
CNN_df = pd.read_csv("rn325/my_analysis/mlbd_dt_models/comparison_study/eval_metrics/CNN_bootstrap_evaluation_metrics.csv")
# c = pd.read_csv("rn325/GNN/outputs/eval/547496/bootstrap_evaluation_metrics")


# alpha = 0.05

# confusion_matrix_arrs = np.array([[np.mean(GNN_df["tn"]), np.mean(GNN_df["fp"])], [np.mean(GNN_df["fn"]), np.mean(GNN_df["tp"])]])


# plt.pcolormesh(confusion_matrix_arrs, cmap='Blues', shading='auto')
# plt.xticks([0.5, 1.5], ['Background', 'Signal'])
# plt.yticks([0.5, 1.5], ['Background', 'Signal'])
# for i in range(2):
#     for j in range(2):
#         if i == j:
#             text_colour = "white"
#         else:
#             text_colour = "black"
#         plt.text(j+0.5, i+0.5, confusion_matrix_arrs[i] [j], ha='center', va='center', color=text_colour, fontsize=20)
# plt.xlabel('Predicted Class')
# plt.ylabel('True Class')
# plt.colorbar(label='Counts')
# plt.tight_layout()
# title = "rn325/GNN/outputs/eval/548900/trial_confusion_matrix.png"
# plt.savefig(title)
# plt.close()

# metrics = df.to_numpy()
# new = metrics.T[1:8]
# metrics = new.T
# names = df.columns
# #[df["auc_score"].to_numpy(),
auc_numpy =  [GNN_df["auc_score"].to_numpy(), CNN_df["auc_score"].to_numpy()]

# # print(names)
# # print(metrics.shape)

plt.boxplot(auc_numpy, labels = ["GNN", "CNN"], vert = False)
plt.grid()
plt.title("Comparison of AUC scores and percentiles for models 1 and 2")
plt.xlabel("AUC score")
plt.savefig("rn325/my_analysis/mlbd_dt_models/comparison_study/boxplot_CNN_GNN.png")


# aucs = df["auc_score"].to_numpy()


# lower_bound = np.percentile(aucs, alpha/2 * 100)
# upper_bound = np.percentile(aucs, (1 - alpha/2) * 100)

# mean_auc = np.mean(aucs)

# print(f"Number of trials: {len(aucs)}")
# print(f"Mean return: {mean_auc:.4f}")
# print(f"95% Confidence Interval: {lower_bound:.4f} to {upper_bound:.4f}")

# plt.boxplot(aucs, bootstrap = True)

# # plt.hist(aucs)
# # plt.axvline(mean_auc, color='r', linestyle='dashed', linewidth=1)
# # plt.axvline(lower_bound, color='k', linestyle='dashed', linewidth=1)
# # plt.axvline(upper_bound, color='k', linestyle='dashed', linewidth=1)
# # min_ylim, max_ylim = plt.ylim()
# # #plt.text(mean_auc*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(mean_auc))
# # plt.text(0.969, 20, 'Mean: {:.2f}'.format(mean_auc))
# # plt.xlabel("AUC value")
# # plt.ylabel("Counts")
# # plt.title("Histogram of AUC values for 100 bootstrapped samples")

# plt.savefig("rn325/GNN/outputs/eval/548900/aucs.png")

# # plt.boxplot(aucs, bootstrap = True)