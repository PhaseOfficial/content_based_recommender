import pandas as pd

# Input file path (current file with potential encoding issues)
input_file = "c:/Users/Nick/Desktop/content_based_recommender/data/expanded_news_dataset.csv"

# Output file path (new file with UTF-8 encoding)
output_file = "c:/Users/Nick/Desktop/content_based_recommender/data/expanded_news_dataset_utf8.csv"

try:
    # Read the file
    df = pd.read_csv(input_file)

    # Save the file with UTF-8 encoding
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"File successfully saved with UTF-8 encoding at: {output_file}")
except Exception as e:
    print(f"Error: {e}")