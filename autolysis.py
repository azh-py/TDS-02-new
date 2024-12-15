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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_narrative_from_api(payload):
    api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}",
        "Content-Type": "application/json"
    }
    response = httpx.post(api_url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()

def load_data(file_path):
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']
    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    if df.empty or df.columns.isnull().any():
        raise ValueError("Invalid or empty CSV file.")
    return df

def create_output_dir(dataset_name):
    output_dir = os.path.join('.', dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_visuals(df, output_dir):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    for column in numeric_columns[:3]:
        plt.figure()
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_wordcloud(df, output_dir):
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    for column in text_columns[:2]:
        text_data = " ".join(df[column].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400).generate(text_data)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {column}')
        plt.savefig(os.path.join(output_dir, f'{column}_wordcloud.png'), dpi=300, bbox_inches='tight')
        plt.close()

def save_summary_stats(df, output_dir):
    stats_file = os.path.join(output_dir, "summary_statistics.txt")
    df.describe(include='all').to_string(open(stats_file, "w"))

def create_prompt(df):
    prompt = f"""You are a data analyst. Analyze the dataset with {len(df)} rows and {len(df.columns)} columns.
    Highlight key trends, outliers, and actionable insights."""
    return prompt

def save_readme(narrative, output_dir):
    readme_file = os.path.join(output_dir, "README.md")
    with open(readme_file, "w") as f:
        f.write(narrative)

def main(file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = create_output_dir(dataset_name)
    df = load_data(file_path)

    save_visuals(df, output_dir)
    generate_wordcloud(df, output_dir)
    save_summary_stats(df, output_dir)

    prompt = create_prompt(df)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    narrative = fetch_narrative_from_api(payload)['choices'][0]['message']['content']
    save_readme(narrative, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    main(sys.argv[1])
