import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

file_paths = ["pwlds_very_weak.csv", "pwlds_weak.csv", "pwlds_average.csv",
              "pwlds_strong.csv", "pwlds_very_strong.csv"]

dfs = []
for i, file in enumerate(file_paths):
    df = pd.read_csv(file)
    df["Strength_Level"] = i
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df["Strength_Level"].value_counts())

df_sampled = df.groupby("Strength_Level", group_keys=False).apply(lambda x: x.sample(n=20000, random_state=42)).reset_index(drop=True)
print(df_sampled["Strength_Level"].value_counts())

X = np.array(df_sampled["Password"].tolist())
Y = np.array(df_sampled["Strength_Level"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")