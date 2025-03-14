import pandas as pd
import string
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset files
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

# Get random 20,000 passwords (from each file)
df_sampled = df.groupby("Strength_Level", group_keys=False).apply(lambda x: x.sample(n=20000, random_state=42)).reset_index(drop=True)

print(df_sampled["Strength_Level"].value_counts())

chars = string.ascii_letters + string.digits + r"!@#$%^&*()/'\:~.<>?{}[]=+_-;|`"
char_to_num = {char: i + 1 for i, char in enumerate(chars)}
char_to_num["<UNK>"] = 0 # Use index 0 for unknown characters

def encode_password(password, max_len=16):
    encoded = [char_to_num.get(char, 0) for char in password]
    encoded = encoded[:max_len]  # Truncate if too long
    encoded += [0] * (max_len - len(encoded))  # Pad if too short
    return encoded

df_sampled["encoded_password"] = df_sampled["Password"].apply(encode_password)

X = np.array(df_sampled["encoded_password"].tolist(), dtype=np.int32)
Y = np.array(df_sampled["Strength_Level"], dtype=np.int64)

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Save data
np.save("output/X_train.npy", X_train)
np.save("output/X_test.npy", X_test)
np.save("output/Y_train.npy", Y_train)
np.save("output/Y_test.npy", Y_test)

print(f"Training Samples: {len(X_train)}, Test Samples: {len(X_test)}")