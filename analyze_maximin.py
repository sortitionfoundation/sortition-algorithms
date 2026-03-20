# coding: utf-8
import csv
import argparse
from argparse import ArgumentParser
from dataclasses import dataclass
from math import ceil, isclose
from pathlib import Path
import time
from typing import Dict, List, Any, NewType, Union, Tuple
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# IMPORT SWAPPED TO MAXIMIN
try:
    from sortition_algorithms.committee_generation.maximin import find_distribution_maximin
except ImportError as e:
    print(f"CRITICAL ERROR: Could not locate maximin.py. Details: {e}")
    exit(1)

AgentId = NewType("AgentId", Any)
FeatureCategory = NewType("FeatureCategory", str)
Feature = NewType("Feature", str)

@dataclass
class FeatureInfo:
    min: int
    max: int
    selected: int = 0
    remaining: int = 0

ProbAllocation = NewType("ProbAllocation", Dict[AgentId, float])

@dataclass
class Instance:
    k: int
    categories: Dict[FeatureCategory, Dict[Feature, FeatureInfo]]
    agents: Dict[AgentId, Dict[FeatureCategory, Feature]]

def read_instance(feature_file: Union[str, Path], pool_file: Union[str, Path], k: int) -> Instance:
    feature_info = {}
    with open(feature_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            category = FeatureCategory(line["category"])
            feature = Feature(line["name"]) 
            if category not in feature_info:
                feature_info[category] = {}
            feature_info[category][feature] = FeatureInfo(min=int(line["min"]), max=int(line["max"]))

    categories = list(feature_info)
    agents = {}
    with open(pool_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, line in enumerate(reader):
            agent_id = AgentId(str(i))
            agents[agent_id] = {category: Feature(line[category]) for category in categories}
            for category in categories:
                feature_info[category][Feature(line[category])].remaining += 1
    return Instance(k=k, categories=feature_info, agents=agents)

def maximin_probabilities(instance: Instance, backend: str = "mip") -> ProbAllocation:
    portfolio, output_probs, _ = find_distribution_maximin(
        instance.categories, 
        instance.agents, 
        instance.k, 
        [], 
        solver_backend=backend
    )
    selection_probs = {agent_id: 0. for agent_id in instance.agents}
    for panel, probability in zip(portfolio, output_probs):
        for agent_id in panel:
            selection_probs[agent_id] += probability
    return ProbAllocation(selection_probs)

@dataclass
class ProbAllocationStats:
    gini: float
    geometric_mean: float
    min: float
    max: float  

def compute_prob_allocation_stats(alloc: ProbAllocation) -> ProbAllocationStats:
    n = len(alloc)
    k = round(sum(alloc.values()))
    sorted_probs = sorted(alloc.values())
    gini = sum((2 * i - n + 1) * prob for i, prob in enumerate(sorted_probs)) / (n * k)
    geometric_mean = gmean([max(p, 1e-10) for p in sorted_probs])
    mini = min(sorted_probs)
    maxi = max(sorted_probs) 
    return ProbAllocationStats(gini=gini, geometric_mean=geometric_mean, min=mini, max=maxi)

def analyze_instance(instance_name: str, instance: Instance, backend: str, num_runs: int = 1):
    Path("analysis").mkdir(exist_ok=True)
    print(f"Running MAXIMIN analysis with {backend} backend ({num_runs} runs)...")
    
    timings = []
    alloc = None
    for i in range(num_runs):
        start_time = time.time()
        alloc = maximin_probabilities(instance, backend)
        timings.append(time.time() - start_time)
        print(f"  Run {i + 1}/{num_runs} complete.")
    
    total_prob = sum(alloc.values())
    is_valid = isclose(total_prob, instance.k, rel_tol=1e-5)
    
    stats = compute_prob_allocation_stats(alloc)
    
    # Plotting (Scatter Strip Plot Aesthetic)
    n = len(alloc)
    probs = list(alloc.values())
    
    plt.figure(figsize=(10, 3)) 
    
    rng = np.random.default_rng(42)
    jitter = 0.22
    y_vals = rng.uniform(-jitter, jitter, size=n)
    
    # Changed dot color to Red for Maximin
    plt.scatter(probs, y_vals, s=18, alpha=0.55, color="tab:red", label="Maximin")
    
    plt.axvline(x=instance.k/n, color='black', linestyle='--', linewidth=1, label=f'Equalized (k/n = {instance.k/n:.4f})')
    plt.axvline(x=stats.min, color='red', linestyle=':', alpha=0.5, label=f'Min: {stats.min:.4f}')
    plt.axvline(x=stats.max, color='orange', linestyle=':', alpha=0.5, label=f'Max: {stats.max:.4f}')
    
    plt.yticks([0], ["Maximin"])
    plt.grid(axis="x", alpha=0.25)
    plt.ylim(-0.6, 0.6)
    plt.title(f"Maximin Selection Probabilities: {instance_name} (n={n}, k={instance.k})")
    plt.xlabel("Selection Probability")
    plt.legend(loc="upper right")
    
    output_path = Path("analysis") / f"{instance_name}_{instance.k}_maximin_plot.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"VALIDATION: {'✅ PASS' if is_valid else '❌ FAIL'} (Sum={total_prob:.2f})")
    print(f"Gini Score: {stats.gini:.4%}")
    print(f"Min/Max Probs: {stats.min:.6f} / {stats.max:.6f}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('instance_name')
    parser.add_argument('panel_size', type=int)
    parser.add_argument('--backend', default='mip')
    args = parser.parse_args()

    data_path = Path("data") / f"{args.instance_name}_{args.panel_size}"
    instance = read_instance(data_path / "categories.csv", data_path / "respondents.csv", args.panel_size)
    analyze_instance(args.instance_name, instance, args.backend)