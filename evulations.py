from sklearn import metrics
import numpy as np

def calculate_evulations(task, preds, labels, nclass = 1):
    if task == "MCP":
        # labels = np.eye(nclass)[labels]
        micro_f1 = metrics.f1_score(labels, np.argmax(preds, axis = 1), average="micro")
        macro_f1 = metrics.f1_score(labels,  np.argmax(preds, axis = 1), average="macro")
        return [micro_f1, macro_f1], ["micro_f1", "macro_f1"]
    elif task == "Regress":
        mae = metrics.mean_absolute_error(labels, preds)
        rmse =np.sqrt(metrics.mean_squared_error(labels, preds))
        return [rmse, mae], ["rmse", "mae"]
    else:
        f1_score = metrics.f1_score(labels, preds > 0.5)
        auc_score = metrics.roc_auc_score(labels, preds)
        return [f1_score, auc_score], ["f1_score", "auc_score"]

