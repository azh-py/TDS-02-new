# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "chardet",
#   "tenacity",
#   "wordcloud",
#   "scikit-learn"
# ]
# ///

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
import chardet
from wordcloud import WordCloud
from tenacity import retry, stop_after_attempt, wait_fixed
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(file_path):
    """Load and validate the CSV file."""
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]
    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    if df.empty:
        raise ValueError("The dataset is empty.")
    logging.info(f"Data loaded successfully from {file_path}")
    return df

def detect_columns(df):
    """Detect numeric and text columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    logging.info(f"Detected numeric columns: {numeric_cols}")
    return numeric_cols

def create_output_directory(name):
    """Create output directory."""
    output_dir = os.path.join(".", name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def visualize_distributions(df, numeric_cols, output_dir):
    """Generate a limited number of visualizations."""
    for column in numeric_cols[:3]:  # Limit to 3 plots
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"), dpi=300)
        plt.close()
        logging.info(f"Saved distribution for {column}")

@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
def fetch_narrative_from_api(payload):
    """Fetch narrative using API with retry mechanism."""
    api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}", "Content-Type": "application/json"}
    response = httpx.post(api_url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

def generate_narrative(df):
    """Generate summary narrative."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": f"Summarize dataset with {len(df)} rows and {len(df.columns)} columns."}]
    }
    result = fetch_narrative_from_api(payload)
    return result["choices"][0]["message"]["content"]

def save_summary_statistics(df, output_dir):
    """Save summary statistics to a text file."""
    stats_path = os.path.join(output_dir, "summary_statistics.txt")
    df.describe(include="all").to_string(stats_path)
    logging.info("Saved summary statistics.")

def save_readme(narrative, output_dir):
    """Save narrative to README."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(narrative)
    logging.info("Saved README.")

def main(file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = create_output_directory(dataset_name)

    df = load_data(file_path)
    numeric_cols = detect_columns(df)

    visualize_distributions(df, numeric_cols, output_dir)
    save_summary_statistics(df, output_dir)

    narrative = generate_narrative(df)
    save_readme(narrative, output_dir)

    logging.info("Script execution completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
