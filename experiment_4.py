import pandas as pd
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/hp/Desktop/ads/loan-prediction/mlflow.db"
)

mlflow.set_experiment("Loan Eligibility Experiment")
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score


# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_csv("loan_dataset_with_target.csv")

X = df.drop("loan_eligibility", axis=1)
y = df["loan_eligibility"]


# ==============================
# 2. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 3. Identify Column Types
# ==============================

categorical_features = [
    "employment_status",
    "property_ownership"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]


# ==============================
# 4. Preprocessing
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# ==============================
# 5. Baseline Models
# ==============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ==============================
# 6. Baseline Training
# ==============================

results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(f"{name}: {accuracy:.4f}")


# ==============================
# 7. Baseline Results
# ==============================

results_df = pd.DataFrame(results)

print("\nBaseline Model Results:")
print(
    results_df
    .sort_values("Accuracy", ascending=False)
    .to_string(index=False)
)


# ==============================
# 8. Random Forest Tuning
# ==============================

print("\n" + "=" * 50)
print("Random Forest Hyperparameter Tuning")
print("=" * 50)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_param_grid = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run(run_name="Tuned Random Forest"):

    rf_grid.fit(X_train, y_train)

    rf_predictions = rf_grid.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    # Log best hyperparameters
    for parameter, value in rf_grid.best_params_.items():
        mlflow.log_param(parameter, value)

    # Log metrics
    mlflow.log_metric("cv_accuracy", rf_grid.best_score_)
    mlflow.log_metric("test_accuracy", rf_accuracy)

    # Log trained model
    mlflow.sklearn.log_model(
        rf_grid.best_estimator_,
        "random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print("Best Random Forest Parameters:")
print(rf_grid.best_params_)

print(f"Best CV Accuracy: {rf_grid.best_score_:.4f}")
print(f"Test Accuracy: {rf_accuracy:.4f}")


# ==============================
# 9. Logistic Regression Tuning
# ==============================

print("\n" + "=" * 50)
print("Logistic Regression Hyperparameter Tuning")
print("=" * 50)

lr_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

lr_param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs", "liblinear"]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

with mlflow.start_run(run_name="Tuned Logistic Regression"):

    lr_grid.fit(X_train, y_train)

    lr_predictions = lr_grid.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    # Log best hyperparameters
    for parameter, value in lr_grid.best_params_.items():
        mlflow.log_param(parameter, value)

    # Log metrics
    mlflow.log_metric("cv_accuracy", lr_grid.best_score_)
    mlflow.log_metric("test_accuracy", lr_accuracy)

    # Log trained model
    mlflow.sklearn.log_model(
        lr_grid.best_estimator_,
        "logistic_regression_model"
    )

print("Best Logistic Regression Parameters:")
print(lr_grid.best_params_)

print(f"Best CV Accuracy: {lr_grid.best_score_:.4f}")
print(f"Test Accuracy: {lr_accuracy:.4f}")


# ==============================
# 10. Tuned Model Comparison
# ==============================

print("\n" + "=" * 50)
print("Tuned Model Comparison")
print("=" * 50)

print(f"Random Forest Test Accuracy:       {rf_accuracy:.4f}")
print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f}")

import joblib

joblib.dump(
    rf_grid.best_estimator_,
    "tuned_random_forest.pkl"
)

joblib.dump(
    lr_grid.best_estimator_,
    "tuned_logistic_regression.pkl"
)

print("\nTrained models saved successfully:")
print("tuned_random_forest.pkl")
print("tuned_logistic_regression.pkl")