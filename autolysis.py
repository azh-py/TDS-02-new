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
#   "statsmodels",
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
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Load the CSV data dynamically and validate."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    if df.empty:
        raise ValueError("CSV is empty.")
    logging.info(f"Data loaded successfully: {file_path}")
    return df

def detect_columns(df):
    """Detect numeric and text columns dynamically."""
    return {
        "numeric": df.select_dtypes(include=['number']).columns.tolist(),
        "text": [col for col in df.columns if df[col].dtype == 'object']
    }

def save_distribution_plots(df, numeric_columns, output_dir, limit=5):
    """Save up to a limited number of distribution plots to optimize runtime."""
    count = 0
    for column in numeric_columns:
        if count >= limit:
            break
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f"Distribution of {column}")
        plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"), dpi=100)
        plt.close()
        logging.info(f"Saved distribution for: {column}")
        count += 1

@retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
def fetch_narrative(df):
    """Fetch AI-generated narrative with reduced timeout."""
    api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"Summarize: {df.describe().to_string()}"}]}
    headers = {"Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}", "Content-Type": "application/json"}

    try:
        response = httpx.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logging.warning(f"API Error: {e}")
        return "Failed to fetch narrative."

def main(file_path):
    start_time = time.time()
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"./{dataset_name}_output"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(file_path)
    columns = detect_columns(df)

    # Save up to 5 distribution plots
    save_distribution_plots(df, columns["numeric"], output_dir, limit=5)

    # Generate and save summary statistics
    stats_file = os.path.join(output_dir, "summary_statistics.txt")
    df.describe(include='all').to_string(stats_file)
    logging.info("Saved summary statistics.")

    # Fetch narrative
    narrative = fetch_narrative(df)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(narrative)
    logging.info("Generated README.md")

    logging.info(f"Script runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
