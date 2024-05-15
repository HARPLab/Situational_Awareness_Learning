import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


run_name = '/home/harpadmin/Situational_Awareness_Learning/wandb/run-20240513_115922-xsjzcrcc/files/'
y_true = np.load(run_name + 'best_val_gt.npy')
y_pred = np.load(run_name + 'best_val_preds.npy')
y_pred_raw = np.load(run_name + 'best_val_raw_preds.npy')

precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
precision1, recall1, thresholds1 = precision_recall_curve(y_true, y_pred_raw)


# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.savefig(run_name + 'precision_recall_curve.png')


# print(accuracy_score(y_true, y_pred))
# print(precision_score(y_true, y_pred))
# print(recall_score(y_true, y_pred))
# print(average_precision_score(y_true, y_pred))
