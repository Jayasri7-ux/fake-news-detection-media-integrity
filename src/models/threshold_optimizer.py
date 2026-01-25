import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

class ThresholdOptimizer:
    def optimize_threshold(self, model, X_val, y_val, metric='f1'):
        """
        Finds the best threshold for a given metric (default F1).
        """
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_val)[:, 1]
        else:
            return 0.5 # Fallback if no probabilities

        precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
        
        if metric == 'f1':
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            best_idx = np.nanargmax(f1_scores)
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            return best_threshold
        
        return 0.5

    def optimize_for_business(self, model, X_val, y_val, cost_fp=1, cost_fn=5):
        """
        Finds threshold that minimizes business cost.
        Cost = (False Positives * cost_fp) + (False Negatives * cost_fn)
        """
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_val)[:, 1]
        else:
            return 0.5

        thresholds = np.linspace(0, 1, 101)
        costs = []

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            cost = (fp * cost_fp) + (fn * cost_fn)
            costs.append(cost)

        best_idx = np.argmin(costs)
        return thresholds[best_idx]
