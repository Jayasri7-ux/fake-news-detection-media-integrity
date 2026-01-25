import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss

class ModelComparator:
    def compare_models(self, models_dict, X_test, y_test):
        """
        Evaluates multiple models and returns a comparison dataframe.
        """
        results = []
        for name, model in models_dict.items():
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_probs)
                ll = log_loss(y_test, y_probs)
            else:
                auc = np.nan
                ll = np.nan
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Score": f1,
                "ROC AUC": auc,
                "Log Loss": ll
            })
            
        return pd.DataFrame(results)

    def generate_report(self, results_df, output_path="model_comparison_report.md"):
        """
        Generates a markdown report from the comparison results.
        """
        markdown = "# Model Comparison Report\n\n"
        
        # Manual markdown table generation to avoid 'tabulate' dependency
        columns = results_df.columns
        markdown += "| " + " | ".join(columns) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        
        for _, row in results_df.iterrows():
            row_str = " | ".join([str(val) for val in row])
            markdown += f"| {row_str} |\n"
            
        markdown += "\n\n## Statistical Analysis\n"
        markdown += "Note: Detailed McNemar's test or T-test would require per-sample predictions.\n"
        
        # Identify best model
        best_model = results_df.loc[results_df['F1 Score'].idxmax()]
        markdown += f"\n**Best Model based on F1 Score:** {best_model['Model']}\n"
        
        with open(output_path, "w") as f:
            f.write(markdown)
            
        print(f"Report generated at {output_path}")
