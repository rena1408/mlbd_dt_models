import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import os

N_BOOT  = 1000
SEED    = 42
np.random.seed(SEED)

output_dir = "rn325/my_analysis/mlbd_dt_models/model_out_data"
os.makedirs(output_dir, exist_ok=True)

# Load scores and labels
# CNN
cnn_scores = np.load('rn325/my_analysis/mlbd_dt_models/this_model/CNN/separate/test_scores_0.npy')
cnn_labels = np.load('rn325/my_analysis/mlbd_dt_models/this_model/CNN/separate/truth_0.npy').astype(int)

# GraphConv
gnn_scores = np.load('rn325/my_analysis/mlbd_dt_models/this_model/GNN/separate/test_scores_0.npy')
gnn_labels = np.load('rn325/my_analysis/mlbd_dt_models/this_model/GNN/separate/truth_0.npy').astype(int)

# Graph Transformer (signed)
tr_scores = np.load('rn325/my_analysis/mlbd_dt_models/this_model/GT/separate/test_scores_0.npy')
tr_labels = np.load('rn325/my_analysis/mlbd_dt_models/this_model/GT/separate/truth_0.npy').astype(int)

models = [
    ('CNN',               cnn_scores, cnn_labels),
    ('GraphConv',         gnn_scores, gnn_labels),
    ('GraphTransformer',  tr_scores,  tr_labels),
]

models = [
    ('GraphConv',          gnn_scores, gnn_labels),
]

results = {}

for name, scores, labels in models:
    print(f"\nBootstrapping {name} (N={N_BOOT})...")
    n = len(scores)

    auc_boot  = []
    acc_boot  = []
    sens_boot = []
    prec_boot = []
    f1_boot = []

    # Calculate the ROC curve

    y_true = labels
    y_pred_prob = scores

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Method 1: Optimal threshold using ROC curve (point 3)
    optimal_idx = np.argmin(np.sqrt(np.square(1-tpr) + np.square(fpr)))
    optimal_threshold_roc = thresholds[optimal_idx]
    print("Optimal threshold using ROC curve:", optimal_threshold_roc)

    for _ in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)
        s   = scores[idx]
        l   = labels[idx]
        auc_boot.append(roc_auc_score(l, s))
        # acc_boot.append(accuracy_score(l, s > 0.5))
        # sens_boot.append(recall_score(l, s > 0.5))
        # prec_boot.append(precision_score(l, s > 0.5))
        # f1_boot.append(f1_score(l, s > 0.5))

        threshold = optimal_threshold_roc
        acc_boot.append(accuracy_score(l, s > threshold))
        sens_boot.append(recall_score(l, s > threshold))
        prec_boot.append(precision_score(l, s > threshold))
        f1_boot.append(f1_score(l, s > threshold))


    auc_mean = np.mean(auc_boot)
    auc_std  = np.std(auc_boot)
    acc_mean = np.mean(acc_boot)
    acc_std  = np.std(acc_boot)

    sens_mean = np.mean(sens_boot)
    sens_std  = np.std(sens_boot)
    prec_mean = np.mean(prec_boot)
    prec_std  = np.std(prec_boot)
    f1_mean = np.mean(f1_boot)
    f1_std  = np.std(f1_boot)

    # 95% CI
    auc_ci_lo = np.percentile(auc_boot, 2.5)
    auc_ci_hi = np.percentile(auc_boot, 97.5)

    results[name] = {
        'auc_mean': auc_mean, 'auc_std': auc_std,
        'auc_ci_lo': auc_ci_lo, 'auc_ci_hi': auc_ci_hi,
        'acc_mean': acc_mean, 'acc_std': acc_std,
        'sens_mean': sens_mean, 'sens_std': sens_std,
        'prec_mean': prec_mean, 'prec_std': prec_std,
        'f1_mean': f1_mean, 'f1_std': f1_std,
    }

    print(f"  AUC = {auc_mean:.4f} ± {auc_std:.4f} (95% CI: [{auc_ci_lo:.4f}, {auc_ci_hi:.4f}])")
    print(f"  Accuracy = {acc_mean:.4f} ± {acc_std:.4f}")


    print(f"  sensitivity = {sens_mean:.4f} ± {sens_std:.4f}")
    print(f"  precision = {prec_mean:.4f} ± {prec_std:.4f}")
    print(f"  F1 = {f1_mean:.4f} ± {f1_std:.4f}")

# Save results
rows = []
for name, r in results.items():
    rows.append({
        'Model':       name,
        'AUC':         f"{r['auc_mean']:.4f}",
        'AUC_std':     f"{r['auc_std']:.4f}",
        'AUC_CI_lo':   f"{r['auc_ci_lo']:.4f}",
        'AUC_CI_hi':   f"{r['auc_ci_hi']:.4f}",
        'Accuracy':    f"{r['acc_mean']:.4f}",
        'Acc_std':     f"{r['acc_std']:.4f}",
        'Sensitivity':    f"{r['sens_mean']:.4f}",
        'sens_std':     f"{r['sens_std']:.4f}",
        'Precision':    f"{r['prec_mean']:.4f}",
        'prec_std':     f"{r['prec_std']:.4f}",
        'F1':    f"{r['f1_mean']:.4f}",
        'f1_std':     f"{r['f1_std']:.4f}",
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_dir + '/bootstrap_results.csv', index=False)

print(f"\nSaved bootstrap_results.csv to {output_dir}")
print(df_out.to_string(index=False))
