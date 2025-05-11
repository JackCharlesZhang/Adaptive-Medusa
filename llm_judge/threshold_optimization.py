"""
This script supports both CMA-ES (via Nevergrad) and Bayesian Optimization (via Optuna).
To switch modes, change USE_CMA = True/False.
--input: Path to the input JSON file containing the data.  
"""
USE_CMA = True  # Set to False to use Optuna instead

import argparse
import json
import random
import numpy as np
from collections import Counter
import pandas as pd
import nevergrad as ng
import optuna


# === Parse args ===
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--costs", required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

costs = list(map(float, args.costs.strip().split(",")))
assert len(costs) >= 6, "Provide at least 6 cost values"

# === Load and split data ===
random.seed(args.seed)
accept_lengths_all, entropies_all, top_probs_all = [], [], []

with open(args.input, "r") as f:
    for line in f:
        entry = json.loads(line)
        for choice in entry.get("choices", []):
            zipped = list(zip(choice["accept_lengths"], choice["entropies"], choice["top_probs"]))
            random.shuffle(zipped)
            for a, e, p in zipped:
                accept_lengths_all.append(a)
                entropies_all.append(e)
                top_probs_all.append(p)

# === 80-20 split ===
data = list(zip(accept_lengths_all, entropies_all, top_probs_all))
random.shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

def predict_length(entropies, top_probs, entropy_thresholds, prob_thresholds):
    predicted_length = 0
    max_depth = min(len(entropies), len(entropy_thresholds))
    for i in range(max_depth):
        if entropies[i] < entropy_thresholds[i] and top_probs[i] > prob_thresholds[i]:
            predicted_length = i + 1
        else:
            break
    return predicted_length

# === CMA-ES optimization objective ===
def evaluate_tokens_per_sec(params, data_subset):
    entropy_thresholds = params[:5]
    prob_thresholds = params[5:]
    total_tokens = 0
    total_time = 0

    for a, e, p in data_subset:
        pred = predict_length(e, p, entropy_thresholds, prob_thresholds)
        earned_tokens = min(pred, a) + 1
        cost = costs[min(pred, 5)]  # clip to length 5
        total_tokens += earned_tokens
        total_time += cost

    return -total_tokens / total_time if total_time > 0 else float("inf")

# === Define Optuna objective ===
def objective(trial):
    entropy_thresholds = [trial.suggest_float(f"entropy_{i}", 1.0, 7.0) for i in range(5)]
    prob_thresholds = [trial.suggest_float(f"prob_{i}", 0.1, 0.95) for i in range(5)]

    total_tokens = 0
    total_time = 0

    for a, e, p in zip(accept_lengths_all, entropies_all, top_probs_all):
        pred = predict_length(e, p, entropy_thresholds, prob_thresholds)
        earned_tokens = min(pred, a) + 1
        cost = costs[min(pred, 5)]  # clip to avoid out-of-bounds
        total_tokens += earned_tokens
        total_time += cost

    if total_time == 0:
        return float("inf")

    tokens_per_sec = total_tokens / total_time
    return -tokens_per_sec  # Optuna minimizes



# === Search space and optimizer ===
'''
instrumentation = ng.p.Instrumentation(
    *[ng.p.Scalar(init=3.0).set_bounds(1.0, 7.0) for _ in range(5)] +  # entropy thresholds
    [ng.p.Scalar(init=0.3).set_bounds(0.1, 0.95) for _ in range(5)]   # prob thresholds
)
'''

instrumentation = ng.p.Instrumentation(
    *[ng.p.Scalar(init=3.0).set_bounds(0.5, 8.0) for _ in range(5)],
    *[ng.p.Scalar(init=0.3).set_bounds(0.05, 0.98) for _ in range(5)]
)

optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=200)
optimizer.parametrization.random_state.seed(args.seed)
# === Run optimization on training set ===
if USE_CMA:
    budget = 4000
    print("Starting CMA-ES optimization...\n")
    best_loss = float("inf")
    best_params = None

    for i in range(budget):
        x = optimizer.ask()
        loss = evaluate_tokens_per_sec(x.args, train_data)
        optimizer.tell(x, loss)

        if loss < best_loss:
            best_loss = loss
            best_params = x.args

        if (i + 1) % 50 == 0 or i == 0:
            print(f"Step {i + 1}/{budget} — Tokens/sec: {-loss:.4f} (Best so far: {-best_loss:.4f})")
else:
    study = optuna.create_study()
    study.optimize(lambda trial: objective(trial), n_trials=1000)
    best_trial = study.best_trial
    best_params = [best_trial.params[f"entropy_{i}"] for i in range(5)] + \
                  [best_trial.params[f"prob_{i}"] for i in range(5)]

# === Final eval on train/test ===
def report_performance(name, data_subset):
    entropy_thresholds = best_params[:5]
    prob_thresholds = best_params[5:]
    preds = [predict_length(e, p, entropy_thresholds, prob_thresholds) for _, e, p in data_subset]
    acc = [a for a, _, _ in data_subset]
    total_tokens = sum(min(p, a) + 1 for p, a in zip(preds, acc))
    total_time = sum(costs[min(p, 5)] for p in preds)
    match = sum(p == a for p, a in zip(preds, acc)) / len(acc)
    tps = total_tokens / total_time
    print(f"\n{name} Set:")
    print(f"  Match Rate: {match * 100:.2f}%")
    print(f"  Tokens/sec: {tps:.3f}")
    return preds, acc

# === Platonic ideals for Train and Test sets
ideal_tokens_train = sum(a + 1 for a, _, _ in train_data)
ideal_time_train = sum(costs[min(a, 5)] for a, _, _ in train_data)
ideal_tps_train = ideal_tokens_train / ideal_time_train

ideal_tokens_test = sum(a + 1 for a, _, _ in test_data)
ideal_time_test = sum(costs[min(a, 5)] for a, _, _ in test_data)
ideal_tps_test = ideal_tokens_test / ideal_time_test

print("\nMax possible throughput (platonic ideal):")
print(f"  Train Max Tokens/sec: {ideal_tps_train:.3f}")
print(f"  Test  Max Tokens/sec: {ideal_tps_test:.3f}")



train_preds, train_acc = report_performance("Train", train_data)
test_preds, test_acc = report_performance("Test", test_data)

# === Distributions on train and test set
def print_distribution(label, true_labels, pred_labels):
    all_lengths = sorted(set(true_labels) | set(pred_labels) | set(range(6)))
    total = len(true_labels)
    df = pd.DataFrame({
        "Accept %": [true_labels.count(k) / total * 100 for k in all_lengths],
        f"Predicted {label} %": [pred_labels.count(k) / total * 100 for k in all_lengths]
    }, index=all_lengths)
    df.index.name = "Length"
    print(f"\n{label} Set Distributions (%):")
    print(df.round(2))


# Print both
print_distribution("Train", train_acc, train_preds)
print_distribution("Test", test_acc, test_preds)

# === Print final best thresholds used ===
print("\nBest Thresholds Used:")
print("  Entropy Thresholds:", [round(x, 4) for x in best_params[:5]])
print("  Probability Thresholds:", [round(x, 4) for x in best_params[5:]])
