import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from pulearn import BaggingPuClassifier # Using BaggingPU for robustness

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = r"C:\Users\Asus\Desktop\XG-PUL"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data")
FEATURES_DIR = os.path.join(BASE_DIR, "data") 
ALL_SEEDS_DIR = os.path.join(BASE_DIR, "data")
TRAIN_SEEDS_DIR = os.path.join(BASE_DIR, "data")


EMBEDDING_SETUPS = {
    'highdim': {
        'file': os.path.join(EMBEDDINGS_DIR, 'PPI_node2vec_embeddings_highdim.csv'),
        'description': 'High Dimension'
    }
}

DISEASE_CODES_TO_EVALUATE = ["C0001973", "C0005586","C0006142","C0009402","C0023893","C0036341","C0376358","C0860207","C3714756", "C0011581"]
RANDOM_STATE = 42
N_FEATURES_TO_SELECT = 150 # Number of top features to select (embedding)

def read_data(path, index_col, error_msg):
    if not os.path.exists(path):
        print(f"  ERROR: {error_msg} not found: {path}")
        return None
    df = pd.read_csv(path)
    if index_col in df.columns:
        df.set_index(index_col, inplace=True)
        df.index = df.index.str.lower()
        return df
    return None

def read_gene_list(path, error_msg):
    if not os.path.exists(path):
        print(f"  ERROR: {error_msg} not found: {path}")
        return None
    genes = pd.read_csv(path, header=None, sep='\s+')[0].str.lower().tolist()
    return genes

# ==============================================================================
# Find Disease Module Function: PU LEARNING AND RANKING FUNCTION
# ==============================================================================

def FindDiseaseModule():
    """
    Trains a PU Learning model with feature selection and generates a ranked
    list of non-seed genes for comparison.
    """
    setup_name = 'highdim'
    setup_config = EMBEDDING_SETUPS[setup_name]
    
    output_ranking_dir = os.path.join(BASE_DIR, "Ranking_result", "FindDiseaseModule_TopologicalFeatures_EmbedddingHighdim")
    os.makedirs(output_ranking_dir, exist_ok=True)

    print("="*80)
    print(f"Generating Gene Rankings using PU Learning and Feature Selection")
    print("="*80)

    embeddings_df = read_data(setup_config['file'], 'gene', "Embedding file")
    if embeddings_df is None: return

    # # 1. read CLEAN Topological Features
    feature_path = os.path.join(FEATURES_DIR, "topological_features.csv")
    features_df = read_data(feature_path, 'gene', "Topological features file")
    if features_df is None: return

    # 2. Combine Data
    combined_df = pd.merge(embeddings_df, features_df, left_index=True, right_index=True, how='inner')
    print(f"Initial data readed: {len(combined_df)} genes with both embeddings and topological features.")

    for disease_code in DISEASE_CODES_TO_EVALUATE:
        print(f"\n-- Processing Disease: {disease_code} --")

        # 3. read Gene Sets for Hold-Out
        train_genes_path = os.path.join(TRAIN_SEEDS_DIR, f"{disease_code}_seed_genes.txt")
        all_genes_path = os.path.join(ALL_SEEDS_DIR, f"{disease_code}_all_seed_genes.txt")
        
        train_pos_genes = read_gene_list(train_genes_path, "Training seed file")
        all_pos_genes = read_gene_list(all_genes_path, "All seed file")

        if train_pos_genes is None or all_pos_genes is None: continue

        # 4. Define Labels (Positive vs. Unlabeled)
        temp_df = combined_df.copy()
        temp_df['class'] = 0
        positive_genes_in_data = list(set(all_pos_genes) & set(temp_df.index))
        temp_df.loc[positive_genes_in_data, 'class'] = 1
        
        y = temp_df['class']
        X = temp_df.drop(columns=['class'])

        # 5. Prepare Train and Prediction Sets
        train_pos_genes_in_data = list(set(train_pos_genes) & set(X.index))
        prediction_genes = X.index.difference(train_pos_genes_in_data).tolist()
        unlabeled_genes = y[y == 0].index.tolist()

        print(f"  Using {len(train_pos_genes_in_data)} positive genes for training.")
        if not train_pos_genes_in_data:
            print("  WARNING: No training genes found. Skipping.")
            continue

        # Training set for feature selection and PU learning
        X_train_full = X.loc[train_pos_genes_in_data + unlabeled_genes]
        y_train_full = y.loc[train_pos_genes_in_data + unlabeled_genes]
        
        # 6. Feature Selection using RandomForest
        print("  Running feature selection with RandomForest...")
        fs_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        fs_model.fit(X_train_full, y_train_full)
        
        importances = fs_model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = feature_importance_df.head(N_FEATURES_TO_SELECT)['feature'].tolist()
        print(f"  Selected {len(top_features)} most important features.")
        
        X_train_selected = X_train_full[top_features]
        X_predict_selected = X.loc[prediction_genes][top_features]

        # 7. Train PU Learning Model
        print("  Training PU Learning model (BaggingPUClassifier)...")
        # Define the base estimator
        estimator = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE)
        
        # Create the PU classifier
        # n_estimators is the number of bootstrap samples to create
        pu_model = BaggingPuClassifier(estimator, n_estimators=15, n_jobs=-1, random_state=RANDOM_STATE)
        pu_model.fit(X_train_selected, y_train_full)

        # 8. Predict and Rank
        print(f"  Predicting scores for {len(X_predict_selected)} non-seed genes...")
        pred_scores = pu_model.predict_proba(X_predict_selected)[:, 1]
        
        ranking_df = pd.DataFrame({
            'gene': X_predict_selected.index,
            'score': pred_scores
        }).sort_values('score', ascending=False)

        # 9. Save Ranking File
        output_filename = f"{disease_code}_ranking_TopologicalFeatures_EmbedddingHighdim.txt"
        output_path = os.path.join(output_ranking_dir, output_filename)
        
        ranking_df['gene'].to_csv(output_path, index=False, header=False)
        print(f"  Ranking file saved to: {output_path}")

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    FindDiseaseModule()

    # FindDiseaseModule_TopologicalFeatures_EmbedddingHighdim