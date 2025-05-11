import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def process_json_file(file_path):
    """Process JSON lines file containing benchmark data."""
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Process each choice in the answer
                for choice in data["choices"]:
                    # Each turn in the MT-Bench has metrics
                    for i in range(len(choice["turns"])):
                        # Extract the metrics
                        idxs = choice["idxs"][i]  # number of decoding steps
                        new_tokens = choice["new_tokens"][i]  # tokens generated
                        wall_time = choice["wall_time"][i]  # time taken
                        
                        # Calculate metrics
                        acceleration_rate = new_tokens / idxs if idxs > 0 else 0
                        tokens_per_second = new_tokens / wall_time if wall_time > 0 else 0
                        
                        results.append({
                            "question_id": data["question_id"],
                            "model_id": data["model_id"],
                            "turn": i + 1,
                            "decoding_steps": idxs,
                            "new_tokens": new_tokens,
                            "wall_time": wall_time,
                            "acceleration_rate": acceleration_rate,
                            "tokens_per_second": tokens_per_second
                        })
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line}")
                continue
    
    return pd.DataFrame(results)


def calculate_metrics(df, baseline_df=None):
    """Calculate aggregate metrics from the processed data."""
    # Group by model_id to calculate metrics
    metrics = df.groupby('model_id').agg({
        'acceleration_rate': 'mean',
        'tokens_per_second': 'mean',
        'decoding_steps': 'sum',
        'new_tokens': 'sum',
        'wall_time': 'sum'
    }).reset_index()
    
    # Add overall acceleration rate calculated from totals
    metrics['overall_acc_rate'] = metrics['new_tokens'] / metrics['decoding_steps']
    
    # If baseline data is provided, calculate overhead and speedup
    if baseline_df is not None:
        # Get baseline metrics
        baseline = baseline_df.groupby('model_id').agg({
            'tokens_per_second': 'mean'
        }).reset_index()
        baseline = baseline.rename(columns={'tokens_per_second': 'baseline_tokens_per_second'})
        
        # Merge with metrics
        metrics = pd.merge(metrics, baseline, on='model_id', how='left')
        
        # Calculate overhead and speedup
        # Overhead = vanilla_tokens_per_step / medusa_tokens_per_step
        # Since tokens_per_step = tokens_per_second * seconds_per_step,
        # and we know acceleration_rate = tokens / steps,
        # we can calculate overhead as:
        metrics['overhead'] = (metrics['tokens_per_second'] / metrics['overall_acc_rate']) / \
                             (metrics['baseline_tokens_per_second'] / 1.0)
        
        # Speedup = Acceleration rate / Overhead
        metrics['speedup'] = metrics['overall_acc_rate'] / metrics['overhead']
        
        # Or directly: speedup = medusa_tokens_per_second / baseline_tokens_per_second
        metrics['speedup_direct'] = metrics['tokens_per_second'] / metrics['baseline_tokens_per_second']
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Calculate Medusa metrics from benchmark results')
    parser.add_argument('--input', required=True, help='Input JSON lines file with benchmark results')
    parser.add_argument('--baseline', help='Baseline JSON lines file for vanilla model (optional)')
    parser.add_argument('--output', help='Output CSV file for results (optional)')
    
    args = parser.parse_args()
    
    print(f"Processing benchmark data from {args.input}")
    df = process_json_file(args.input)
    
    # If baseline is provided
    baseline_df = None
    if args.baseline:
        print(f"Processing baseline data from {args.baseline}")
        baseline_df = process_json_file(args.baseline)
    
    # Calculate metrics
    metrics = calculate_metrics(df, baseline_df)
    
    # Print results
    print("\n===== Metrics =====")
    if baseline_df is not None:
        print(metrics[['model_id', 'overall_acc_rate', 'overhead', 'speedup', 'tokens_per_second']])
    else:
        print(metrics[['model_id', 'overall_acc_rate', 'tokens_per_second']])
    
    # Save results if output is specified
    if args.output:
        metrics.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    


if __name__ == '__main__':
    main()
