import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score

# Load dataset
df = pd.read_csv("/Users/jessicalarissa/Desktop/alzheimers_prediction_dataset.csv")

# Fix column name for label
target_column = "Alzheimer’s Diagnosis"
y = df[target_column]
X = df.drop(columns=[target_column])

# Identify feature types
numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Add cognitive deviation feature
healthy_mean = df[df[target_column] == "No"]["Cognitive Test Score"].mean()
X["Cognitive Test Deviation from Healthy Mean"] = (df["Cognitive Test Score"] - healthy_mean)
numeric_cols.append("Cognitive Test Deviation from Healthy Mean")

# Preprocessing pipelines
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_cols),
    ("cat", cat_transformer, categorical_cols)
])

# Pipeline factory
def make_pipeline(model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selector", SelectKBest(mutual_info_classif, k=50)),
        ("classifier", model)
    ])

# Define models and parameters
models = {
    "Logistic Regression": (LogisticRegression(class_weight = "balanced", max_iter=1000), {
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ['saga']
    }),
    "Decision Tree": (DecisionTreeClassifier(class_weight = "balanced"), {
        "classifier__max_depth": [5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10]
    }),
    "kNN": (KNeighborsClassifier(weights="distance"), {
        'classifier__n_neighbors': [5, 55, 99, 389, 767]
    })
}

# Define scorers
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score, pos_label="Yes", zero_division=0),
    "precision": make_scorer(precision_score, pos_label="Yes", zero_division=0),
    "recall": make_scorer(recall_score, pos_label="Yes", zero_division=0)
}

# Define CV strategies
outer_cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

# Run nested cross-validation
# Run manual nested CV
results = []
for model_name, (model, param_grid) in models.items():
    print(f"Tuning {model_name}...")
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV GridSearch
        pipe = make_pipeline(model)
        grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring=scoring, refit=False, n_jobs=-1)
        grid.fit(X_train, y_train)

        # Progress update
        print(f"Finished outer fold {fold} for {model_name}")

        # Record all tuning results for this outer fold
        cv_df = pd.DataFrame(grid.cv_results_)
        cv_df["Outer Fold"] = fold
        cv_df["Model"] = model_name
        results.append(cv_df)

# Combine and display
tuning_summary = pd.concat(results, ignore_index=True)
summary_display = tuning_summary[[
    "Model", "Outer Fold", "params",
    "mean_test_accuracy", "mean_test_f1",
    "mean_test_precision", "mean_test_recall"
]]

# Detect all hyperparameter columns
param_cols = [col for col in tuning_summary.columns if col.startswith("param_")]
metric_cols = [col for col in tuning_summary.columns if col.startswith("mean_test_")]

# Group by model + param combo and average across folds
avg_by_param = tuning_summary.groupby(["Model"] + param_cols)[metric_cols].mean().reset_index()
std_by_param = tuning_summary.groupby(["Model"] + param_cols)[metric_cols].std().reset_index()

# Printing results: 
# 1. summary_display: Scores per fold for each hyperparameter combination, each classifier.
# 2. avg_by_param: Average scores across folds for each hyperparameter combination, each classifier.
# 3. std_by_param: Standard deviation for each average scores across folds. 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print("\n==========  All Model Tuning Results ========== \n")
print(summary_display)
print("\n========== Average Scores Across Outer Folds Per Hyperparameter Combination ==========\n")
print(avg_by_param)
print("\n========== Standard Deviation Across Outer Folds Per Hyperparameter Combination ==========\n")
print(std_by_param)


print("Tuning complete!")