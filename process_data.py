import pandas as pd
import string
from sklearn.model_selection import train_test_split
import numpy as np

file_paths = ["PWLDS/pwlds_very_weak.csv", "PWLDS/pwlds_weak.csv", "PWLDS/pwlds_average.csv",
              "PWLDS/pwlds_strong.csv", "PWLDS/pwlds_very_strong.csv"]

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

chars = string.ascii_letters + string.digits + "!@#$%^&*()"
char_to_num = {char : i + 1 for i, char in enumerate(chars)}

def encode_password(password, max_len=16):
    encoded = [char_to_num.get(char, 0) for char in password]
    return encoded + [0] * (max_len - len(password)) #pad

df_sampled["encoded_password"] = df_sampled["Password"].apply(encode_password)

# split dataset columns
X = np.array(df_sampled["Password"].tolist())
Y = np.array(df_sampled["Strength_Level"])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) #split training and testing data

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("Y_train.npy", Y_train)
np.save("Y_test.npy", Y_test)

print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")