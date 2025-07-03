import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv('mnist_train.csv', header=None)

# Split train into train+val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Load test set (if separate)
test_df = pd.read_csv('mnist_test.csv', header=None)

# Save them
train_df.to_csv('./data/mnist_split_train.csv', index=False, header=False)
val_df.to_csv('./data/mnist_split_val.csv', index=False, header=False)
test_df.to_csv('./data/mnist_split_test.csv', index=False, header=False)
print("Dataset split completed and saved as mnist_split_train.csv, mnist_split_val.csv, and mnist_split_test.csv")