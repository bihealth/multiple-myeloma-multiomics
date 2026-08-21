#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from argparse import ArgumentParser

from mofapy2.run.entry_point import entry_point
import mofax as mfx

parser = ArgumentParser()
parser.add_argument('-n', dest='n', default=10, type=int, help="number of factors [10]")
parser.add_argument('--samples', dest='samples', default=None,
                    help="path to text file with one sample name per line to subset (default: use all)")
parser.add_argument('--label', dest='label', default=None,
                    help="label for output files (default: no label, files named by factor count only)")
args = parser.parse_args()

DATA_DIR   = "mofa2_input"
OUTPUT_DIR = "mofa2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CNV_FILE     = os.path.join(DATA_DIR, "CNV_data_processed.csv")
RNA_FILE     = os.path.join(DATA_DIR, "RNA_data_processed.csv")
PROTEIN_FILE = os.path.join(DATA_DIR, "protein_data_processed.csv")
PHOSPHO_FILE = os.path.join(DATA_DIR, "phospho_data_processed.csv")

# Model options
NUM_FACTORS  = args.n     # upper bound; inactive factors pruned automatically
SEED         = 42
MAX_ITER     = 1000
CONVERGENCE  = "medium"   # "fast" / "medium" / "slow"
SCALE_VIEWS  = True       # recommended when views have very different variances

RUN_TAG = f"{args.label}_n{NUM_FACTORS}" if args.label else f"n{NUM_FACTORS}"
MODEL_FILE   = os.path.join(OUTPUT_DIR, f"mofa2_model_{RUN_TAG}.hdf5")

# Feature subsampling by variance (set to None to keep all)
TOP_CNV_FEATURES     = None
TOP_RNA_FEATURES     = None
TOP_PROTEIN_FEATURES = None
TOP_PHOSPHO_FEATURES = None

def load_csv(path, name, orient="features_x_samples"):
    """Load CSV -> DataFrame samples x features."""
    df = pd.read_csv(path, index_col=0)
    if orient == "features_x_samples":
        df = df.T
    df = df.astype(np.float32)
    print(f"  [{name}] {df.shape[0]} samples x {df.shape[1]} features")
    return df


def top_var_features(df, n, name):
    """Keep n highest-variance features; None = keep all."""
    if n is None or n >= df.shape[1]:
        return df
    top = df.var(axis=0).nlargest(n).index
    print(f"  [{name}] subsampled {df.shape[1]} -> {n} features by variance")
    return df[top]


def align_samples(*dfs):
    common = sorted(set.intersection(*[set(df.index) for df in dfs]))
    print(f"  Shared samples: {len(common)}")
    return [df.loc[common] for df in dfs], common


cnv_df     = load_csv(CNV_FILE,     "CNV",     orient="samples_x_features")
rna_df     = load_csv(RNA_FILE,     "RNA",     orient="samples_x_features")
protein_df = load_csv(PROTEIN_FILE, "Protein", orient="samples_x_features")
phospho_df = load_csv(PHOSPHO_FILE, "Phospho", orient="samples_x_features")

print("\nAligning samples ...")
(cnv_df, rna_df, protein_df, phospho_df), common_samples = align_samples(
    cnv_df, rna_df, protein_df, phospho_df
)

if args.samples is not None:
    with open(args.samples) as fh:
        sample_subset = [s.strip() for s in fh if s.strip()]
    missing = set(sample_subset) - set(common_samples)
    if missing:
        print(f"  WARNING: {len(missing)} requested samples not in data and will be skipped: {sorted(missing)}")
    sample_subset = [s for s in sample_subset if s in set(common_samples)]
    print(f"  Using {len(sample_subset)} of {len(common_samples)} samples (from {args.samples})")
    cnv_df     = cnv_df.loc[sample_subset]
    rna_df     = rna_df.loc[sample_subset]
    protein_df = protein_df.loc[sample_subset]
    phospho_df = phospho_df.loc[sample_subset]
    common_samples = sample_subset

print("\nSubsampling features by variance ...")
rna_df     = top_var_features(rna_df,     TOP_RNA_FEATURES,     "RNA")
protein_df = top_var_features(protein_df, TOP_PROTEIN_FEATURES, "Protein")
phospho_df = top_var_features(phospho_df, TOP_PHOSPHO_FEATURES, "Phospho")
cnv_df     = top_var_features(cnv_df,     TOP_CNV_FEATURES,     "CNV")


rna_df.columns = 'RNA_' + rna_df.columns
protein_df.columns = 'protein_' + protein_df.columns
phospho_df.columns = 'phospho_' + phospho_df.columns

print("\n" + "=" * 60)
print("Training MOFA2 model ...")
print("=" * 60)

ent = entry_point()

views_data = [
    [cnv_df.values.copy()],      # view 0, group 0
    [rna_df.values.copy()],      # view 1, group 0
    [protein_df.values.copy()],  # view 2, group 0
    [phospho_df.values.copy()],  # view 3, group 0
]

ent.set_data_matrix(
    views_data,
    likelihoods  = ["gaussian", "gaussian", "gaussian", "gaussian"],
    views_names  = ["CNV", "RNA", "Protein", "Phospho"],
    samples_names= [common_samples],   # list of lists: one per group
    features_names = [
        list(cnv_df.columns),
        list(rna_df.columns),
        list(protein_df.columns),
        list(phospho_df.columns),
    ],
) 
# Data options
ent.set_data_options(
    scale_views=SCALE_VIEWS,   # scale each view to unit variance
    scale_groups=False,
)

# Model options
ent.set_model_options(
    factors=NUM_FACTORS,
    spikeslab_weights=True,    # feature-wise sparsity (recommended)
    spikeslab_factors=False,   # sample-wise sparsity (not needed here)
    ard_weights=True,          # view-wise ARD prior (prunes irrelevant factors per view)
    ard_factors=True,          # group-wise ARD prior
)

# Training options
ent.set_train_options(
    iter=MAX_ITER,
    convergence_mode=CONVERGENCE,
    seed=SEED,
    verbose=False,
    gpu_mode=True
)

ent.build()
ent.run()
ent.save(MODEL_FILE)
print(f"\n  Model saved -> {MODEL_FILE}")
print(f"  Reload: model = mfx.mofa_model('{MODEL_FILE}')")

print("\n" + "=" * 60)
print("Downstream analysis (mofax) ...")
print("=" * 60)

model = mfx.mofa_model(MODEL_FILE)
print(model)

# Factor scores: samples x factors
Z = model.get_factors(df=True)          # DataFrame: samples x factors

n_factors = Z.shape[1]
factor_names = [f"Factor{i+1}" for i in range(n_factors)]

Z.index = common_samples                 # restore sample names if needed
Z.columns = factor_names
Z.to_csv(os.path.join(OUTPUT_DIR, f"factor_scores_{RUN_TAG}.csv"))

# Weights: per view, features x factors
for view in ["CNV", "RNA", "Protein", "Phospho"]:
    try:
        W = model.get_weights(views=view, df=True)
        W.columns = factor_names[:W.shape[1]]
        fname = f"weights_{view.lower()}_{RUN_TAG}.csv"
        W.to_csv(os.path.join(OUTPUT_DIR, fname))
    except Exception as e:
        print(f"  WARNING [{view}]: {e}")

print("\nComputing variance explained ...")

r2_df = model.get_r2(per_factor=True)   # DataFrame: factor x view

# Plot: heatmap factors x views
# pivot to factors x views for heatmap
r2_pivot = r2_df.pivot_table(index="Factor", columns="View", values="R2")
# preserve factor order (Factor1, Factor2, ... not lexicographic)
r2_pivot = r2_pivot.loc[
    sorted(r2_pivot.index, key=lambda x: int(x.replace("Factor", "")))
]

r2_pivot.to_csv(os.path.join(OUTPUT_DIR, f"variance_explained_{RUN_TAG}.csv"))

print("\nDone.")
