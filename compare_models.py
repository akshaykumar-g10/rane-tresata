import pandas as pd

from semantic_model import load_model as load_lr, predict_column_type as predict_col_lr
from semantic_model_svm import load_model as load_svm, predict_column_type as predict_col_svm

df_phone = pd.read_csv("data/phone.csv")
df_company = pd.read_csv("data/company.csv")

lr_model = load_lr()
svm_model = load_svm()

print("=== PHONE COLUMN ===")
print("LR:", predict_col_lr(df_phone, "number", model=lr_model))
print("SVM:", predict_col_svm(df_phone, "number", model=svm_model))

print("\n=== COMPANY COLUMN ===")
print("LR:", predict_col_lr(df_company, "company", model=lr_model))
print("SVM:", predict_col_svm(df_company, "company", model=svm_model))
