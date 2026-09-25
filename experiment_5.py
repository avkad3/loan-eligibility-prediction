import os
import warnings
import logging

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "loan_dataset_with_target.csv"
MODEL_FILE = "tuned_random_forest.pkl"

OUTPUT_DIR = "experiment_5_outputs"

TARGET_COLUMN = "loan_eligibility"

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_sensitive_attribute(df):
    """
    Automatically identify a likely sensitive attribute.
    """

    candidates = [
        "gender",
        "sex",
        "race",
        "ethnicity",
        "age",
        "marital_status",
        "marital",
    ]

    lower_columns = {
        column.lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lower_columns:
            return lower_columns[candidate]

    return None


def save_text_report(text, filename):
    path = os.path.join(OUTPUT_DIR, filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    return path


# ============================================================
# 1. LOAD DATA AND MODEL
# ============================================================

logging.info("Loading dataset...")

df = pd.read_csv(DATA_FILE)

logging.info("Dataset shape: %s", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found in the dataset."
    )

logging.info("Loading trained Random Forest model...")

model = joblib.load(MODEL_FILE)

logging.info("Model loaded successfully.")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

logging.info("Features: %s", list(X.columns))

logging.info("Generating predictions...")

predictions = model.predict(X)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X)[:, 1]
else:
    probabilities = None

logging.info(
    "Model accuracy on the available dataset: %.4f",
    accuracy_score(y, predictions)
)


# ============================================================
# 3. SHAP GLOBAL EXPLANATION
# ============================================================

logging.info("Generating SHAP explanations...")

try:

    # Extract preprocessing and Random Forest from pipeline
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    X_transformed = preprocessor.transform(X)

    # Obtain transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert sparse matrix to dense if necessary
    if hasattr(X_transformed, "toarray"):
        X_transformed_dense = X_transformed.toarray()
    else:
        X_transformed_dense = X_transformed

    shap_explainer = shap.TreeExplainer(classifier)

    shap_values = shap_explainer.shap_values(
        X_transformed_dense
    )

    # Handle SHAP versions where classification output is a list
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values_positive = shap_values[1]
        else:
            shap_values_positive = shap_values[0]
    else:
        shap_values_positive = shap_values

    # --------------------------------------------------------
    # SHAP SUMMARY PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values_positive,
        X_transformed_dense,
        feature_names=feature_names,
        show=False
    )

    plt.title("SHAP Global Feature Importance")

    plt.tight_layout()

    shap_summary_path = os.path.join(
        OUTPUT_DIR,
        "shap_summary.png"
    )

    plt.savefig(
        shap_summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logging.info(
        "SHAP summary saved: %s",
        shap_summary_path
    )

    # --------------------------------------------------------
    # SHAP BAR IMPORTANCE
    # --------------------------------------------------------

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values_positive,
        X_transformed_dense,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    plt.title("SHAP Feature Importance")

    plt.tight_layout()

    shap_bar_path = os.path.join(
        OUTPUT_DIR,
        "shap_feature_importance.png"
    )

    plt.savefig(
        shap_bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SHAP DEPENDENCE PLOT
    # --------------------------------------------------------

    mean_importance = np.abs(
        shap_values_positive
    ).mean(axis=0)

    most_important_index = np.argmax(
        mean_importance
    )

    most_important_feature = feature_names[
        most_important_index
    ]

    plt.figure(figsize=(10, 7))

    shap.dependence_plot(
        most_important_feature,
        shap_values_positive,
        X_transformed_dense,
        feature_names=feature_names,
        show=False
    )

    plt.title(
        f"SHAP Dependence Plot: {most_important_feature}"
    )

    plt.tight_layout()

    shap_dependence_path = os.path.join(
        OUTPUT_DIR,
        "shap_dependence.png"
    )

    plt.savefig(
        shap_dependence_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE SHAP IMPORTANCE TABLE
    # --------------------------------------------------------

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": mean_importance
    })

    shap_importance = shap_importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    shap_importance.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "shap_feature_importance.csv"
        ),
        index=False
    )

    logging.info(
        "Most influential feature according to SHAP: %s",
        most_important_feature
    )

except Exception as error:

    logging.exception(
        "SHAP analysis failed: %s",
        error
    )

    shap_values_positive = None
    feature_names = None


# ============================================================
# 4. LIME LOCAL EXPLANATION
# ============================================================

logging.info("Generating LIME local explanation...")

try:

    # Convert categorical columns to strings
    X_lime = X.copy()

    categorical_columns = X_lime.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # LIME works better when categorical variables
    # are represented consistently.
    for column in categorical_columns:
        X_lime[column] = X_lime[column].astype(str)

    # Create encoded representation for LIME
    X_lime_encoded = pd.get_dummies(
        X_lime,
        drop_first=False
    )

    lime_feature_names = X_lime_encoded.columns.tolist()

    lime_data = X_lime_encoded.astype(float).values

    def lime_predict(data):

        incoming = pd.DataFrame(
            data,
            columns=lime_feature_names
        )

        # Reconstruct original feature dataframe
        reconstructed = pd.DataFrame(
            columns=X.columns,
            index=range(len(incoming))
        )

        for column in X.columns:

            if column in incoming.columns:

                reconstructed[column] = incoming[column]

            else:

                matching_columns = [
                    c for c in incoming.columns
                    if c.startswith(column + "_")
                ]

                if matching_columns:

                    reconstructed[column] = (
                        incoming[matching_columns]
                        .idxmax(axis=1)
                        .str.replace(
                            column + "_",
                            "",
                            regex=False
                        )
                    )

                else:

                    reconstructed[column] = 0

        return model.predict_proba(
            reconstructed
        )

    lime_explainer = LimeTabularExplainer(
        lime_data,
        feature_names=lime_feature_names,
        class_names=[
            "Not Eligible",
            "Eligible"
        ],
        mode="classification",
        random_state=RANDOM_STATE
    )

    # Explain first sample
    sample_index = 0

    explanation = lime_explainer.explain_instance(
        lime_data[sample_index],
        lime_predict,
        num_features=10
    )

    lime_path = os.path.join(
        OUTPUT_DIR,
        "lime_explanation.html"
    )

    explanation.save_to_file(lime_path)

    # Save textual explanation
    lime_text_path = os.path.join(
        OUTPUT_DIR,
        "lime_explanation.txt"
    )

    with open(
        lime_text_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LIME Local Explanation\n"
            "=======================\n\n"
        )

        file.write(
            f"Sample index: {sample_index}\n"
        )

        file.write(
            f"Actual class: {y.iloc[sample_index]}\n"
        )

        file.write(
            f"Predicted class: "
            f"{predictions[sample_index]}\n\n"
        )

        file.write(
            "Feature contributions:\n\n"
        )

        for feature, weight in explanation.as_list():

            file.write(
                f"{feature}: {weight:.6f}\n"
            )

    logging.info(
        "LIME explanation saved: %s",
        lime_path
    )

except Exception as error:

    logging.exception(
        "LIME analysis failed: %s",
        error
    )


# ============================================================
# 5. FAIRNESS AUDIT
# ============================================================

logging.info("Starting fairness audit...")

sensitive_attribute = find_sensitive_attribute(df)

fairness_report = []

fairness_report.append(
    "FAIRNESS AUDIT REPORT\n"
    "=====================\n"
)

fairness_report.append(
    f"Dataset: {DATA_FILE}\n"
)

fairness_report.append(
    f"Model: {MODEL_FILE}\n"
)

fairness_report.append(
    f"Target: {TARGET_COLUMN}\n\n"
)

if sensitive_attribute is None:

    message = (
        "No common sensitive attribute was found.\n"
        "Expected columns include: gender, sex, race, "
        "ethnicity, age, or marital_status.\n"
        "Fairness metrics were therefore not calculated."
    )

    fairness_report.append(message)

    logging.warning(message)

else:

    logging.info(
        "Sensitive attribute selected: %s",
        sensitive_attribute
    )

    sensitive_values = df[sensitive_attribute]

    # --------------------------------------------------------
    # DEMOGRAPHIC PARITY DIFFERENCE
    # --------------------------------------------------------

    dp_difference = demographic_parity_difference(
        y,
        predictions,
        sensitive_features=sensitive_values
    )

    # --------------------------------------------------------
    # EQUALIZED ODDS DIFFERENCE
    # --------------------------------------------------------

    eo_difference = equalized_odds_difference(
        y,
        predictions,
        sensitive_features=sensitive_values
    )

    fairness_report.append(
        f"Sensitive attribute: "
        f"{sensitive_attribute}\n\n"
    )

    fairness_report.append(
        f"Demographic Parity Difference: "
        f"{dp_difference:.6f}\n"
    )

    fairness_report.append(
        f"Equalized Odds Difference: "
        f"{eo_difference:.6f}\n\n"
    )

    # --------------------------------------------------------
    # GROUP-WISE PERFORMANCE
    # --------------------------------------------------------

    group_rows = []

    for group in sensitive_values.dropna().unique():

        mask = sensitive_values == group

        y_group = y[mask]

        pred_group = predictions[mask]

        group_accuracy = accuracy_score(
            y_group,
            pred_group
        )

        group_precision = precision_score(
            y_group,
            pred_group,
            zero_division=0
        )

        group_recall = recall_score(
            y_group,
            pred_group,
            zero_division=0
        )

        group_f1 = f1_score(
            y_group,
            pred_group,
            zero_division=0
        )

        positive_rate = (
            pred_group.mean()
        )

        group_rows.append({
            "group": group,
            "sample_count": len(y_group),
            "accuracy": group_accuracy,
            "precision": group_precision,
            "recall": group_recall,
            "f1_score": group_f1,
            "positive_prediction_rate": positive_rate
        })

    group_results = pd.DataFrame(group_rows)

    group_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "fairness_group_metrics.csv"
        ),
        index=False
    )

    fairness_report.append(
        "Group-wise Performance\n"
        "-----------------------\n"
    )

    fairness_report.append(
        group_results.to_string(
            index=False
        )
    )

    fairness_report.append("\n\n")

    # --------------------------------------------------------
    # FAIRNESS INTERPRETATION
    # --------------------------------------------------------

    fairness_report.append(
        "Interpretation\n"
        "--------------\n"
    )

    fairness_report.append(
        "Demographic parity difference measures "
        "differences in positive prediction rates "
        "between groups.\n\n"
    )

    fairness_report.append(
        "Equalized odds difference measures differences "
        "in prediction error rates across groups.\n\n"
    )

    fairness_report.append(
        "Values closer to zero indicate smaller measured "
        "group disparities for the selected metric.\n"
    )

    fairness_report.append(
        "These metrics should be interpreted together "
        "with the underlying group sizes and task context.\n"
    )

    # --------------------------------------------------------
    # FAIRNESS VISUALIZATION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        group_results["group"].astype(str),
        group_results["positive_prediction_rate"]
    )

    plt.xlabel(sensitive_attribute)
    plt.ylabel("Positive Prediction Rate")
    plt.title(
        "Positive Prediction Rate by Group"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    fairness_plot_path = os.path.join(
        OUTPUT_DIR,
        "fairness_positive_prediction_rate.png"
    )

    plt.savefig(
        fairness_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 6. SAVE FAIRNESS REPORT
# ============================================================

fairness_report_text = "\n".join(
    fairness_report
)

fairness_report_path = save_text_report(
    fairness_report_text,
    "fairness_audit_report.txt"
)

logging.info(
    "Fairness report saved: %s",
    fairness_report_path
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

summary = f"""
============================================================
EXPERIMENT 5 COMPLETE
============================================================

Model:
{MODEL_FILE}

Dataset:
{DATA_FILE}

Target:
{TARGET_COLUMN}

Accuracy on available dataset:
{accuracy_score(y, predictions):.4f}

SHAP outputs:
- shap_summary.png
- shap_feature_importance.png
- shap_dependence.png
- shap_feature_importance.csv

LIME outputs:
- lime_explanation.html
- lime_explanation.txt

Fairness outputs:
- fairness_audit_report.txt
- fairness_group_metrics.csv
- fairness_positive_prediction_rate.png

Sensitive attribute:
{sensitive_attribute if sensitive_attribute else "Not detected"}

============================================================
CONCLUSION
============================================================

SHAP was used to identify the features that contributed most
strongly to the Random Forest predictions at a global level.

LIME was used to explain an individual model prediction and
show which feature conditions contributed to that prediction.

Fairlearn was used to evaluate prediction disparities across
the selected sensitive attribute using demographic parity
difference and equalized odds difference.

If measurable disparities are observed, possible mitigation
strategies include reweighting or resampling during
pre-processing, fairness-constrained model training, or
post-processing threshold adjustment.

Because the loan eligibility target in this project was
engineered from financial variables, the explainability and
fairness findings should be interpreted as an audit of this
specific experimental dataset and target-generation rule,
rather than as evidence about a real-world lending system.
"""

print(summary)

save_text_report(
    summary,
    "experiment_5_summary.txt"
)

logging.info(
    "Experiment 5 completed successfully."
)