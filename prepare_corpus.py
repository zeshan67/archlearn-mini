import os
import pandas as pd


def main():
    # 1. Path to your raw CSV file
    # If your file is named differently, change it here:
    raw_path = os.path.join("data", "raw", "candidate_list.csv")

    # 2. Path where cleaned corpus will be saved
    out_path = os.path.join("data", "interim", "corpus_v1.csv")

    print(f"Loading raw data from: {raw_path}")

    # 3. Check the file exists
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Could not find the file: {raw_path}")

    # 4. Read the CSV file into a DataFrame
    df = pd.read_csv(raw_path)

    # 5. Make sure there is an 'abstract' column
    if "abstract" not in df.columns:
        raise ValueError("Your CSV must have a column named 'abstract'.")

    # 6. If there is no 'title' column, create one filled with 'Untitled'
    if "title" not in df.columns:
        df["title"] = "Untitled"

    # 7. Remove rows where 'abstract' is missing
    before = len(df)
    df = df.dropna(subset=["abstract"])
    after = len(df)
    print(f"Removed {before - after} rows without abstracts. Remaining: {after}")

    # 8. Strip leading/trailing spaces from text
    df["abstract"] = df["abstract"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()

    # 9. Ensure the 'data/interim' folder exists
    os.makedirs(os.path.join("data", "interim"), exist_ok=True)

    # 10. Save the cleaned corpus
    df.to_csv(out_path, index=False)
    print(f"Clean corpus saved to: {out_path}")


if __name__ == "__main__":
    main()
