import pandas as pd

input_file = 'final_test_2022.csv' 
output_file = 'final_test_2022_no_did.csv'

df = pd.read_csv(input_file)

# Remove the 'Division_ID' column
if 'Division_ID' in df.columns:
    df = df.drop(columns=['Division_ID'])
    print("Column 'Division_ID' removed.")
else:
    print("Column 'Division_ID' not found in the dataset.")

# Save the cleaned dataset to a new CSV file
df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to {output_file}")
