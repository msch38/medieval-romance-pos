# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 18:27:30 2025

@author: esteb
"""

import os
import re
from pathlib import Path

def extract_config_from_path(file_path):
    """
    Extract configuration details from the file path.
    Expected format: MELT/model_name/promptingname-dataset/tagging_decodingname_dataset_promptingname_modelname_5/classification_report.txt
    """
    parts = Path(file_path).parts
    
    # Extract model name (e.g., 'phi4', 'gemma3')
    model = parts[1]
    
    # Extract prompting and dataset (e.g., 'zero-naf', 'few-cat')
    prompting_dataset = parts[2].split('-')
    prompting = prompting_dataset[0]  # 'zero' or 'few'
    dataset = prompting_dataset[1]    # 'naf', 'cat', 'chauliac'
    
    # Extract decoding strategy from folder name (e.g., 'tagging_k50_naf_zero_phi4_5')
    folder_name = parts[3]
    
    # Parse decoding strategy - it's after 'tagging_' and before the next '_'
    decoding_match = re.search(r'tagging_([^_]+)_', folder_name)
    decoding = decoding_match.group(1) if decoding_match else "unknown"
    
    return model, prompting, dataset, decoding

def find_all_classification_reports(base_path):
    """
    Find all classification_report.txt files in the directory structure.
    """
    reports = []
    base_path = Path(base_path)
    
    # Walk through all subdirectories
    for report_file in base_path.rglob("classification_report.txt"):
        try:
            # Extract configuration from path
            relative_path = report_file.relative_to(base_path.parent)
            model, prompting, dataset, decoding = extract_config_from_path(relative_path)
            
            # Read the report content
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            reports.append({
                'file_path': str(relative_path),
                'model': model,
                'prompting': prompting,
                'dataset': dataset,
                'decoding': decoding,
                'content': content,
                'config_id': f"{model}_{prompting}_{dataset}_{decoding}"
            })
            
        except Exception as e:
            print(f"Error processing {report_file}: {e}")
    
    return reports

def combine_reports_to_file(reports, output_file):
    """
    Combine all reports into a single text file with delimiters.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# POS Tagging Classification Reports Analysis\n")
        f.write(f"# Total reports found: {len(reports)}\n")
        f.write(f"# Generated automatically from MELT directory structure\n\n")
        
        for i, report in enumerate(reports, 1):
            f.write("===REPORT_START===\n")
            f.write(f"CONFIG: {report['config_id']}\n")
            f.write(f"MODEL: {report['model']}\n")
            f.write(f"PROMPTING: {report['prompting']}\n")
            f.write(f"DATASET: {report['dataset']}\n")
            f.write(f"DECODING: {report['decoding']}\n")
            f.write(f"FILE_PATH: {report['file_path']}\n")
            f.write(f"REPORT_NUMBER: {i}/{len(reports)}\n")
            f.write("---CONTENT_START---\n")
            f.write(report['content'])
            f.write("\n---CONTENT_END---\n")
            f.write("===REPORT_END===\n\n")

def main():
    # Set the base path
    base_path = r"C:\Users\esteb\Downloads\MELT"
    output_file = r"C:\Users\esteb\Downloads\MELT_analysis\combined_classification_reports.txt"
    
    print("Starting extraction of classification reports...")
    print(f"Base path: {base_path}")
    
    # Check if base path exists
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        return
    
    # Find all reports
    print("Scanning for classification reports...")
    reports = find_all_classification_reports(base_path)
    
    print(f"Found {len(reports)} classification reports")
    
    # Show summary by configuration
    models = set(r['model'] for r in reports)
    promptings = set(r['prompting'] for r in reports)
    datasets = set(r['dataset'] for r in reports)
    decodings = set(r['decoding'] for r in reports)
    
    print(f"\nSummary:")
    print(f"Models found: {sorted(models)}")
    print(f"Prompting strategies: {sorted(promptings)}")
    print(f"Datasets: {sorted(datasets)}")
    print(f"Decoding strategies: {sorted(decodings)}")
    
    # Show counts by model
    print(f"\nCounts by model:")
    for model in sorted(models):
        count = len([r for r in reports if r['model'] == model])
        print(f"  {model}: {count} reports")
    
    # Show any missing combinations (assuming all should exist)
    expected_total = len(models) * len(promptings) * len(datasets) * len(decodings)
    if len(reports) < expected_total:
        print(f"\nNote: Expected {expected_total} reports, but found {len(reports)}")
    
    # Combine reports
    print(f"\nCombining reports into {output_file}...")
    combine_reports_to_file(reports, output_file)
    
    print(f"Successfully created {output_file}")
    print(f"File contains {len(reports)} classification reports ready for analysis")

if __name__ == "__main__":
    main()