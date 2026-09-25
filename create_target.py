import pandas as pd

# Load the original dataset
df = pd.read_csv("modified_loan_dataset.csv")

# Create loan eligibility target
df["loan_eligibility"] = (
    (df["credit_score"] >= 650)
    & (df["total_dti"] <= 0.60)
    & (df["annual_income"] >= 50000)
).astype(int)

# Save the new dataset
df.to_csv("loan_dataset_with_target.csv", index=False)

# Display results
print("Target distribution:")
print(df["loan_eligibility"].value_counts())

print("\nTarget percentages:")
print(df["loan_eligibility"].value_counts(normalize=True) * 100)

print("\nNew dataset shape:")
print(df.shape)