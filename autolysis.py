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
from sklearn.tree import DecisionTreeClassifier
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_message(message):
    logging.info(message)

def load_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        log_message(f"Data loaded successfully from {file_path}")
        validate_csv(df)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def validate_csv(df):
    if df.empty:
        raise ValueError("The CSV file is empty.")
    if df.columns.isnull().any():
        raise ValueError("The CSV file has missing column names.")
    log_message("CSV validation passed.")

def detect_columns(df):
    detected = {
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "text_columns": [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 30]
    }
    log_message(f"Detected Columns: {detected}")
    return detected

def create_output_directory(dataset_name):
    output_dir = os.path.join(".", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    log_message(f"Created output directory: {output_dir}")
    return output_dir

def visualize_distributions(df, numeric_columns, output_dir):
    for column in numeric_columns[:5]:  # Limit the number of plots to reduce runtime
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        output_file = os.path.join(output_dir, f'{column}_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log_message(f"Saved distribution plot for {column} as {output_file}")

def generate_wordcloud(df, text_columns, output_dir):
    for column in text_columns[:2]:  # Limit the number of word clouds to reduce runtime
        text_data = " ".join(df[column].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {column}')
        output_file = os.path.join(output_dir, f'{column}_wordcloud.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        log_message(f"Saved word cloud for {column} as {output_file}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_narrative_from_api(payload):
    api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['AIPROXY_TOKEN']}",
        "Content-Type": "application/json"
    }
    response = httpx.post(api_url, headers=headers, json=payload, timeout=10)  # Reduced timeout
    response.raise_for_status()
    log_message(f"API Response: {response.json()}")
    return response.json()

def create_prompt(df):
    prompt = f"""
    You are a data analyst. Provide concise insights based on:
    - Number of Rows: {len(df)}
    - Number of Columns: {len(df.columns)}
    - Key Data Types: {list(df.dtypes.unique())}
    Identify key trends, anomalies, and actionable findings.
    """
    log_message("Generated concise API prompt.")
    return prompt

def generate_narrative(df):
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": create_prompt(df)}
        ]
    }
    try:
        result = fetch_narrative_from_api(payload)
        return result['choices'][0]['message']['content']
    except KeyError as e:
        log_message(f"KeyError in API response: {e}")
        return "Narrative generation failed. Please check the API response."

def save_summary_statistics(df, output_dir):
    stats_file = os.path.join(output_dir, "summary_statistics.txt")
    with open(stats_file, "w") as f:
        f.write(df.describe(include='all').to_string())
    log_message(f"Saved summary statistics as {stats_file}")

def save_readme(narrative, output_dir):
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(narrative)
    log_message(f"Saved README.md at {readme_path}")

def verify_output_files(output_dir):
    required_files = glob.glob(os.path.join(output_dir, "*.png"))
    if not required_files:
        raise RuntimeError(f"No output files found in {output_dir}. Check for issues.")
    if not os.path.exists(os.path.join(output_dir, "README.md")):
        raise RuntimeError(f"README.md missing in {output_dir}.")
    log_message("Output verification successful.")

def main(file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = create_output_directory(dataset_name)

    df = load_data(file_path)
    columns = detect_columns(df)

    visualize_distributions(df, columns['numeric_columns'], output_dir)
    generate_wordcloud(df, columns['text_columns'], output_dir)

    save_summary_statistics(df, output_dir)
    narrative = generate_narrative(df)
    save_readme(narrative, output_dir)
    verify_output_files(output_dir)
    log_message("Analysis completed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
