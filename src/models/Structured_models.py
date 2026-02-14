import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_models(X_train, X_test, y_train, y_test):

    # -------------------------
    # Define Models
    # -------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = {}
    confusion_matrices = {}

    # -------------------------
    # Train & Evaluate
    # -------------------------
    for name, model in models.items():

        print(f"\n🔹 Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = [acc, prec, rec, f1]
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    # -------------------------
    # Convert to DataFrame
    # -------------------------
    results_df = pd.DataFrame(results,
                              index=["Accuracy", "Precision", "Recall", "F1 Score"]).T

    best_model_name = results_df["F1 Score"].idxmax()
    best_model = models[best_model_name]

    print("\n🏆 Best Model:", best_model_name)
    print(results_df)

    # -------------------------
    # Create Dashboard
    # -------------------------
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Model Evaluation Dashboard", fontsize=22, fontweight='bold')

    axes = axes.flatten()

    # Plot confusion matrices
    for i, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=axes[i])
        axes[i].set_title(f"Confusion Matrix — {name}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # -------------------------
    # Performance Comparison Plot
    # -------------------------
    performance_ax = axes[len(confusion_matrices)]

    results_df.plot(kind='bar',
                    ax=performance_ax)

    performance_ax.set_title("Model Performance Comparison")
    performance_ax.set_ylim(0, 1)
    performance_ax.set_ylabel("Score")
    performance_ax.tick_params(axis='x', rotation=30)

    # Hide last unused subplot if any
    if len(confusion_matrices) + 1 < len(axes):
        axes[-1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # -------------------------
    # Save Dashboard
    # -------------------------
    # Save as High Quality PNG
    png_path = Path.cwd() / "results" / "Model_Evaluation_Dashboard.png"
    plt.savefig(png_path,
                dpi=300,
                bbox_inches='tight')

    plt.show()

    return results_df, best_model, best_model_name
