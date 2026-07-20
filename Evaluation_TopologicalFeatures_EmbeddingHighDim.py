
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 1. CONFIGURATION & PATHS (فقط مسیرها و لیست‌ها اصلاح شدند)
# ==============================================================================
BASE_DIR = r"C:\Users\Asus\Desktop\XG-PUL"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data")
FEATURES_DIR = os.path.join(BASE_DIR, "data") 
ALL_SEEDS_DIR = os.path.join(BASE_DIR, "data")
TRAIN_SEEDS_DIR = os.path.join(BASE_DIR, "data")
RANKINGS_DIR = os.path.join(BASE_DIR, "Rankings") 

## outpust result for evaluation
OUTPUT = os.path.join(BASE_DIR, "Results")
OUTPUT_SUMMARY_DIR = os.path.join(OUTPUT, "Summary_Results_Per_K_F1_Evaluation_XG-PUL - TopologicalFeaturesEmbedddingHighdim")
OUTPUT_PLOTS_DIR = os.path.join(OUTPUT, "Comparison_Plots_F1_Evaluation_XG-PUL - TopologicalFeaturesEmbedddingHighdim")

for folder in [OUTPUT_SUMMARY_DIR, OUTPUT_PLOTS_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

# disease names and code
DISEASE_CODES = {
    "C0006142_Malignant_neoplasm_of_breast": "C0006142", 
    "C0009402_Colorectal_Carcinoma": "C0009402",
    "C0023893_Liver_Cirrhosis_Experimental": "C0023893", 
    "C0036341_Schizophrenia": "C0036341",
    "C0376358_Malignant_neoplasm_of_prostate": "C0376358", 
    "C0001973_Alcoholic_Intoxication_Chronic": "C0001973",
    "C0011581_Depressive_disorder": "C0011581", 
    "C0860207_Drug_Induced_Liver_Disease": "C0860207",
    "C3714756_Intellectual_Disability": "C3714756", 
    "C0005586_Bipolar_Disorder": "C0005586"
}

# نام مدل‌ها (نام پوشه‌های داخل Rankings)
METHODS = ["DIAMOnD", "MCL", "RWR","XGDAG - GNNExplainer","XGDAG - GraphSVX","XG-PUL"]
# مقادیر K برای ران‌های متوالی
RATIOS_TO_VALIDATE = [25, 50, 100, 200, 500, 750, 1000, 1500, 2000, 2500, 3000]

# ==============================================================================
# 2. ORIGINAL CALCULATION LOGIC (حفظ منطق فایل ارسالی شما)
# ==============================================================================
def calculate_f_measure(ranked_genes, true_genes, k):
    """    Calculates Precision, Recall, and F-measure at a specific K.
    """
    # تبدیل به مجموعه برای محاسبه اشتراک (مطابق کد شما)
    top_k = set(ranked_genes[:k])
    true_set = set(true_genes)
    
    hits = len(top_k.intersection(true_set))
    
    precision = hits / k
    recall = hits / len(true_set) if len(true_set) > 0 else 0
    
    if (precision + recall) > 0:
        f_measure = (2 * precision * recall) / (precision + recall)
    else:
        f_measure = 0
        
    return precision, recall, f_measure

# ==============================================================================
# 3. PROCESSING LOOP (مدیریت فایل‌ها و محاسبات)
# ==============================================================================
plot_results = []
k_buckets = {k: [] for k in RATIOS_TO_VALIDATE}

all_seed_files = os.listdir(ALL_SEEDS_DIR) if os.path.exists(ALL_SEEDS_DIR) else []

print("Operation start...")

for disease_full, disease_id in DISEASE_CODES.items():
    # پیدا کردن فایل Seed
    target_seed_file = next((f for f in all_seed_files if disease_id in f and f.endswith('.txt')), None)
    if not target_seed_file: continue
    
    seed_path = os.path.join(ALL_SEEDS_DIR, target_seed_file)
    true_genes = pd.read_csv(seed_path, header=None, sep=r'\s+', engine='python')[0].values

    for method in METHODS:
        method_dir = os.path.join(RANKINGS_DIR, method)
        
        # جستجوی فایل خروجی متد برای بیماری خاص
        filename = f"{method.lower()}_output_{disease_full}.txt"
        filepath = os.path.join(method_dir, filename)
        
        if not os.path.exists(filepath) and os.path.isdir(method_dir):
            for f in os.listdir(method_dir):
                if disease_id in f and f.endswith('.txt'):
                    filepath = os.path.join(method_dir, f)
                    break

        if os.path.exists(filepath):
            try:
                ranked_genes = pd.read_csv(filepath, header=None, engine='python')[0].values
                
                for k in RATIOS_TO_VALIDATE:
                    p, r, f = calculate_f_measure(ranked_genes, true_genes, k)
                    
                    # ذخیره برای نمودار
                    plot_results.append({'Disease': disease_id, 'Method': method, 'K': k, 'F1': f})
                    
                    # ذخیره برای فایل CSV هر K
                    k_buckets[k].append({
                        'Disease_ID': disease_id,
                        'Method': method,
                        'Precision': p,
                        'Recall': r,
                        'F-measure': f
                    })
            except Exception as e:
                print(f"خطا در پردازش فایل {disease_id} مدل {method}: {e}")

# ==============================================================================
# 4. EXPORTING RESULTS (ذخیره فایل‌ها و رسم نمودارها)
# ==============================================================================

# ۱. ذخیره فایل‌های CSV تفکیک شده بر اساس K
for k, rows in k_buckets.items():
    if rows:
        df_k = pd.DataFrame(rows)
        df_k.to_csv(os.path.join(OUTPUT_SUMMARY_DIR, f"Evaluation_XG-PUL - TopologicalFeaturesEmbedddingHighdim_at_K_{k}.csv"), index=False)

# ۲. رسم نمودار مقایسه‌ای روند F-measure برای هر بیماری
df_plot = pd.DataFrame(plot_results)
if not df_plot.empty:
    for d_id in df_plot['Disease'].unique():
        plt.figure(figsize=(10, 6))
        subset = df_plot[df_plot['Disease'] == d_id]
        sns.lineplot(data=subset, x='K', y='F1', hue='Method', marker='o')
        
        plt.title(f'F-measure Trend Comparison - {d_id}')
        plt.grid(True, linestyle='--')
        plt.xticks(RATIOS_TO_VALIDATE, rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f"F1_Curve_{d_id}.png"))
        plt.close()

print(f"\n[It was succesfull]")