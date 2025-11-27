import pandas as pd
from semantic_model import load_model, predict_cell_type, predict_column_type

model = load_model()

print(predict_cell_type("+91 9876543210", model=model))
print(predict_cell_type("Tresata Pvt Ltd", model=model))
print(predict_cell_type("India", model=model))
print(predict_cell_type("2024-11-27", model=model))

df = pd.read_csv("data/phone.csv")
label, probs = predict_column_type(df, df.columns[0], model=model)
print(label, probs)
