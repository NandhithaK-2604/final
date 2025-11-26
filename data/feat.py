import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, VarianceThreshold, RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(r"C:\Users\nnand\Downloads\italian_audio_features_full.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_numeric = X.select_dtypes(include=[np.number])

# -----------------------------
# Preprocessing
# -----------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# -----------------------------
# Feature selection methods
# -----------------------------
X_pos = np.maximum(0, X_scaled)  # Chi2 requires non-negative

chi2_scores = chi2(X_pos, y)[0]
anova_scores = f_classif(X_scaled, y)[0]
mi_scores = mutual_info_classif(X_scaled, y)
variance_scores = VarianceThreshold(threshold=0.01).fit(X_scaled).variances_

rfe_scores = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=5).fit(X_scaled, y).ranking_
sfs_scores = SequentialFeatureSelector(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=5).fit(X_scaled, y).get_support().astype(int)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y)
rf_scores = rf_model.feature_importances_

perm_scores = permutation_importance(rf_model, X_test, y_test, n_repeats=20, random_state=42).importances_mean

lasso_scores = np.abs(LogisticRegression(penalty='l1', solver='saga', max_iter=5000, random_state=42).fit(X_scaled, y).coef_[0])
elastic_scores = np.abs(LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=5000, random_state=42).fit(X_scaled, y).coef_[0])
ridge_scores = np.abs(LogisticRegression(penalty='l2', solver='saga', max_iter=5000, random_state=42).fit(X_scaled, y).coef_[0])
logreg_scores = np.abs(LogisticRegression(penalty=None, max_iter=2000).fit(X_scaled, y).coef_[0])
dt_scores = DecisionTreeClassifier(random_state=42).fit(X_scaled, y).feature_importances_

# -----------------------------
# Create feature scores dataframe
# -----------------------------
feature_scores_df = pd.DataFrame({
    'Feature': X_numeric.columns,
    'Chi2': chi2_scores,
    'ANOVA': anova_scores,
    'MutualInfo': mi_scores,
    'Variance': variance_scores,
    'RFE_Rank': rfe_scores,
    'SFS': sfs_scores,
    'RandomForest': rf_scores,
    'Permutation': perm_scores,
    'LASSO': lasso_scores,
    'ElasticNet': elastic_scores,
    'Ridge': ridge_scores,
    'LogReg': logreg_scores,
    'DecisionTree': dt_scores
})

# -----------------------------
# Normalize for consensus
# -----------------------------
score_cols = feature_scores_df.columns[1:]
norm_df = feature_scores_df.copy()
for col in score_cols:
    if 'Rank' in col:  # invert rank
        norm_df[col] = 1 / (norm_df[col] + 1e-6)
    norm_df[col] = (norm_df[col] - norm_df[col].min()) / (norm_df[col].max() - norm_df[col].min())

# Consensus score = mean across all algorithms
norm_df['Consensus'] = norm_df[score_cols].mean(axis=1)

# Best algorithm per feature
norm_df['Best_Algorithm'] = norm_df[score_cols].idxmax(axis=1)
best_features_per_algo = {col: norm_df.loc[norm_df[col].idxmax(), 'Feature'] for col in score_cols}
best_features_df = pd.DataFrame(list(best_features_per_algo.items()), columns=['Algorithm', 'Best_Feature'])

# Rank all algorithms per feature
algo_ranks_df = norm_df[['Feature'] + list(score_cols)].copy()
algo_ranks_df['Algorithms_Ranked'] = algo_ranks_df[score_cols].apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)

# Sort by consensus
norm_df_sorted = norm_df.sort_values(by='Consensus', ascending=False)

# Per-algorithm sorted features
sorted_dfs = []
for col in score_cols:
    sorted_features = feature_scores_df[['Feature', col]].sort_values(by=col, ascending=False).reset_index(drop=True)
    sorted_features = sorted_features.rename(columns={'Feature': f'{col}_Feature', col: f'{col}_Score'})
    sorted_dfs.append(sorted_features[[f'{col}_Feature']])
algo_sorted_features_df = pd.concat(sorted_dfs, axis=1)

# -----------------------------
# Save feature selection results
# -----------------------------
norm_df_sorted.to_csv(r"C:\Users\nnand\Downloads\feature_significance_with_consensus1.csv", index=False)
best_features_df.to_csv(r"C:\Users\nnand\Downloads\best_features_per_algorithm.csv", index=False)
algo_ranks_df.to_csv(r"C:\Users\nnand\Downloads\feature_algorithm_ranking.csv", index=False)
algo_sorted_features_df.to_csv(r"C:\Users\nnand\Downloads\algorithm_sorted_features.csv", index=False)

print("Feature selection with consensus saved!")

# -----------------------------
# Train Random Forest on top 75 features per algorithm
# -----------------------------
accuracy_dict = {}
top_k = 75

for col in score_cols:
    # Select top 75 features
    top_features = feature_scores_df[['Feature', col]].sort_values(by=col, ascending=False).head(top_k)['Feature'].tolist()
    top_feature_indices = [X_numeric.columns.get_loc(f) for f in top_features]
    X_top = X_scaled[:, top_feature_indices]

    # Train/test split
    X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y, test_size=0.3, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_top, y_train_top)

    # Predict
    y_train_pred = rf.predict(X_train_top)
    y_test_pred = rf.predict(X_test_top)

    # Store accuracies
    accuracy_dict[col] = {'Train_Accuracy': accuracy_score(y_train_top, y_train_pred),
                          'Test_Accuracy': accuracy_score(y_test_top, y_test_pred)}

# Accuracy dataframe
accuracy_df = pd.DataFrame(accuracy_dict).T.reset_index().rename(columns={'index': 'Algorithm'})
accuracy_df.to_csv(r"C:\Users\nnand\Downloads\rf_top75_feature_accuracy.csv", index=False)
print("Random Forest accuracies with top 75 features saved!")
print(accuracy_df)
