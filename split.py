import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Baca anotasi
df = pd.read_csv("cluster.csv")   # pastikan ini versi yang 2785 baris

# 2. Split train + temp (val+test), 80% : 20%
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# 3. Split temp jadi val + test (masing-masing 10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=42
)

print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

# 4. Simpan jadi 3 CSV
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
