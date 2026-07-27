
from config import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = PATH_TO_DATASETS
RANKINGS_DIR = PATH_TO_RANKINGS
OUTPUT_DIR = PATH_TO_RESULTS
FIGURES_DIR = PATH_TO_FIGURES

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# K values for evaluation
K_VALUES = [25, 50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000]

# Disease names and codes
DISEASES = {
    "C0006142_Malignant_neoplasm_of_breast": "C0006142",
    "C0009402_Colorectal_Carcinoma": "C0009402",
    "C0023893_Liver_Cirrhosis": "C0023893",
    "C0036341_Schizophrenia": "C0036341",
    "C0376358_Malignant_neoplasm_of_prostate": "C0376358",
    "C0001973_Chronic_Alcoholic_Intoxication": "C0001973",
    "C0011581_Depressive_disorder": "C0011581",
    "C0860207_Drug-Induced_Liver_Disease": "C0860207",
    "C3714756_Intellectual_Disability": "C3714756",
    "C0005586_Bipolar_Disorder": "C0005586",
}


METHODS = ["XGDAG - GNNExplainer", "XGDAG - GraphSVX", "DIAMOnD", "MCL", "RWR", "XG-PUL"]

# Color palette
CUSTOM_COLORS = {
    "XG-PUL": "#57D397",
    "MCL": "#21109E",
    "RWR": "#9370DB",
    "XGDAG - GNNExplainer": "#0072B2",
    "XGDAG - GraphSVX": "#FF8D29",
    "DIAMOnD": "#808080",
}



def read_genes(path):
    df = pd.read_csv(path, header=None, sep=r"\s+", engine="python")
    return [g.upper() for g in df[0].astype(str).tolist()]

def precision_recall_f1_at_k(ranked_genes, test_pos, k):
    top_k = ranked_genes[:k]
    hits = len(set(top_k) & set(test_pos))
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(test_pos) if len(test_pos) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def auc_pr_from_points(precisions, recalls):
    df = pd.DataFrame({"recall": recalls, "precision": precisions})
    df = df.groupby("recall", as_index=False)["precision"].max().sort_values("recall")
    if len(df) < 2:
        return 0.0
    return float(auc(df["recall"], df["precision"]))

def find_ranking_file(method_dir, disease_code):
    for f in os.listdir(method_dir):
        if disease_code in f:
            return os.path.join(method_dir, f)
    return None

def calculate_auc_metrics(ranked_genes, test_pos):
    """
    Calculate ROC-AUC and PR-AUC over the complete ranking.
    No top-K cutoff and no decision threshold are applied.
    """

    test_pos = set(test_pos)

    y_true = []
    y_score = []

    n = len(ranked_genes)

    for rank, gene in enumerate(ranked_genes):

        # true disease association
        y_true.append(1 if gene in test_pos else 0)

        # ranking score
        # higher rank -> higher score
        y_score.append(n - rank)


    y_true = np.array(y_true)
    y_score = np.array(y_score)


    # -------------------------
    # ROC-AUC
    # -------------------------
    roc_auc = roc_auc_score(
        y_true,
        y_score
    )


    # -------------------------
    # PR curve and PR-AUC
    # -------------------------
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_score
    )


    pr_auc = auc(
        recall,
        precision
    )


    # -------------------------
    # General AUC name
    # (same as ROC-AUC)
    # -------------------------
    auc_value = roc_auc


    return auc_value, roc_auc, pr_auc



# ============================================================
# MAIN EVALUATION
# ============================================================
all_rows = []
summary_rows = []

for disease_name, disease_code in DISEASES.items():
    # seeds
    train_seed_file = os.path.join(DATA_DIR, f"{disease_code}_seed_genes.txt")
    all_seed_file = os.path.join(DATA_DIR, f"{disease_name}_all_seed_genes.txt")

    train_seeds = read_genes(train_seed_file)
    all_seeds = read_genes(all_seed_file)
    test_seeds = list(set(all_seeds) - set(train_seeds))

    for method in METHODS:
        method_dir = os.path.join(RANKINGS_DIR, method)
        if not os.path.exists(method_dir):
            continue

        ranking_file = find_ranking_file(method_dir, disease_code)
        if not ranking_file:
            continue

        ranked_genes = read_genes(ranking_file)
        ranked_genes = [g for g in ranked_genes if g not in set(train_seeds)]


        # roc_auc, pr_auc = calculate_auc_metrics(ranked_genes, test_seeds)
        auc_value, roc_auc, pr_auc = calculate_auc_metrics(ranked_genes,test_seeds)
        # -----------------------------------------------------------------

        precisions, recalls, f1s = [], [], []
        for k in K_VALUES:
            p, r, f1 = precision_recall_f1_at_k(ranked_genes, test_seeds, k)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            
            all_rows.append({
                "Disease": disease_name,
                "Method": method,
                "K": k,
                "Precision": p,
                "Recall": r,
                "F1": f1,
            })

        summary_rows.append({

            "Disease": disease_name,

            "Method": method,

            "AUC": auc_value,

            "ROC_AUC": roc_auc,

            "PR_AUC": pr_auc,

            "F1@500": f1s[K_VALUES.index(500)]

        })
# ============================================================
# SAVE RESULTS
# ============================================================
df_long = pd.DataFrame(all_rows)
df_sum = pd.DataFrame(summary_rows)

df_long.to_csv(os.path.join(OUTPUT_DIR, "detailed_metrics.csv"), index=False)
df_sum.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
# =============================================================================
# 6. BAR CHARTS (per disease) - Optimized for non-compressed layout
# =============================================================================
sns.set(style="whitegrid")

METRICS = ["F1", "Precision", "Recall"]

for k in K_VALUES:
    for metric in METRICS:
        k_df = df_long[df_long["K"] == k]
        if k_df.empty:
            continue

        k_df = k_df.copy()
        k_df["Disease_Display"] = k_df["Disease"].apply(lambda x: x.split("_", 1)[1].replace("_", " "))

        plt.figure(figsize=(18, 8)) 
        
        ax = sns.barplot(
            data=k_df,
            x="Disease_Display",
            y=metric,
            hue="Method",
            palette=CUSTOM_COLORS,
        )

        plt.title(f"{metric}-score comparison at K={k}", fontsize=16, fontweight="bold")
        plt.xlabel("Disease name", fontsize=14)
        plt.ylabel(f"{metric}-score", fontsize=14)
        
        plt.xticks(rotation=45, ha="right", fontsize=11)
        
        plt.tight_layout(pad=2.0)

        plt.savefig(os.path.join(FIGURES_DIR, f"BarChart_{metric}_K{k}.png"), dpi=300)
        plt.close()
# =============================================================================
# 7. LINE PLOT (per disease) - Fixed compression for K=25 to 50 using Log Scale
# =============================================================================
print("\n📊 Generating Line Plots for all metrics...")
METRICS = ["F1", "Precision", "Recall"]

for DISEASE_NAME, CODE in DISEASES.items():
    full_disease_name = DISEASE_NAME.split("_", 1)[1].replace("_", " ")
    disease_df = df_long[df_long["Disease"] == DISEASE_NAME]
    
    if disease_df.empty:
        continue

    for metric in METRICS:
        plt.figure(figsize=(12, 7), dpi=300)
        sns.set_style("whitegrid", {"grid.linestyle": "--"})

        sns.lineplot(
            data=disease_df,
            x="K",
            y=metric,
            hue="Method",
            palette=CUSTOM_COLORS,
            style="Method",
            markers=True,
            dashes=False,
            linewidth=2,
        )

        # Highlight your main method
        for line in plt.gca().get_lines():
            if line.get_label() == "XG-PUL":
                line.set_linewidth(4)
                line.set_zorder(10)

        plt.xscale('log')
        plt.xticks(K_VALUES, labels=[str(k) for k in K_VALUES], rotation=45)

        plt.title(f"{metric}-score comparison: {full_disease_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Top K genes (Log Scale)", fontsize=12)
        plt.ylabel(f"{metric}-score", fontsize=12)

        plt.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.tight_layout()

        file_friendly_name = full_disease_name.replace(" ", "_")
        plot_name = f"{metric}_Comparison_{file_friendly_name}.png"
        plt.savefig(os.path.join(FIGURES_DIR, plot_name), bbox_inches="tight")
        plt.close()
        print(f"   ✅ Saved: {plot_name}")
# =============================================================================
# 8. OPTIONAL: Mean F1@K across diseases (if you want global plot too)
# =============================================================================
mean_df = df_long.groupby(["Method", "K"])["F1"].mean().reset_index()

plt.figure(figsize=(12, 7), dpi=300)
sns.set_style("whitegrid", {"grid.linestyle": "--"})

sns.lineplot(
    data=mean_df,
    x="K",
    y="F1",
    hue="Method",
    palette=CUSTOM_COLORS,
    style="Method",
    markers=True,
    dashes=False,
    linewidth=2,
)

for line in plt.gca().get_lines():
    if line.get_label() == "XG-PUL":
        line.set_linewidth(4)
        line.set_zorder(10)

plt.title("Mean F1-score across diseases", fontsize=14, fontweight="bold")
plt.xlabel("Top K genes", fontsize=12)
plt.ylabel("F1-score", fontsize=12)
plt.xticks(K_VALUES )
plt.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig(os.path.join(FIGURES_DIR, "F1_Comparison_MeanAcrossDiseases.png"), bbox_inches="tight")
plt.close()
print("   ✅ Saved: F1_Comparison_MeanAcrossDiseases.png")
print("✅ Evaluation complete. Results saved in:", FIGURES_DIR)






print("\n📊 Generating Mean F1@K Plot (Averaged across all diseases)...")

plt.figure(figsize=(12, 7), dpi=300)
sns.set_style("whitegrid", {"grid.linestyle": "--"})

mean_data_list = []
for method in METHODS:
    sub = df_long[df_long["Method"] == method]
    if sub.empty:
        continue



output_path = os.path.join(FIGURES_DIR, "mean_f1_k_refined.png")
plt.savefig(output_path, bbox_inches="tight")
plt.close()

print(f"   ✅ Saved: mean_f1_k_refined.png")
