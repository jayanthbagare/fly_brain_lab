"""
STAGE 1 — run this on YOUR machine (or Google Colab), not in a sandbox.

Goal: fetch a subset of the male CNS (fly brain) connectome and turn each
neuron into a row of connectivity-derived features, with its known
neurotransmitter type as the label we'll later try to predict.

SETUP (one-time):
1. pip install neuprint-python
2. Go to https://neuprint.janelia.org, sign in (Google account works),
   then Account -> "Auth Token" to get your API token.
3. Paste that token below where it says YOUR_TOKEN_HERE.

WHY THIS TASK:
Neurotransmitter type is annotated in the dataset already (so we have
ground truth), but it's *not* a connectivity feature itself — it's a
biological property. The question we're asking: does a neuron's wiring
pattern alone carry a statistical signature of what neurotransmitter it
uses? That's a real, non-trivial question, and small enough to answer
in an afternoon.
"""

import os
from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("NEUR_TOKEN")
DATASET = "male-cns:v1.0"

client = Client("https://neuprint.janelia.org", dataset=DATASET, token=TOKEN)

# ---- Step A: pick a manageable region instead of the whole brain (~150k neurons) ----
# Central complex is a well-studied, self-contained circuit (~thousands of neurons,
# not hundreds of thousands) with well-annotated neurotransmitter predictions.
region_criteria = NC(
    inputRois=["EB", "FB", "PB", "NO"], roi_req="any", regex=False
)  # central complex ROIs
# roi_req="any" = neuron touches AT LEAST ONE of these regions.
# (default is "all", which wrongly demands a neuron touch every region at once —
# that's why the first run only found 20 neurons.)
neurons, roi_counts = fetch_neurons(region_criteria)

print(f"Fetched {len(neurons)} neurons in the central complex.")
print(neurons["predictedNt"].value_counts())  # sanity check: label distribution

# Drop neurons with no predicted neurotransmitter (can't use as label)
neurons = neurons.dropna(subset=["predictedNt"])
neuron_ids = neurons["bodyId"].tolist()

# ---- Step B: pull full adjacency (who connects to whom, and how strongly) ----
# fetch_adjacencies returns TWO tables: (neuron_df, connection_df).
# We already have neuron_df (it's `neurons`, from fetch_neurons above), so we
# discard the first return value here and keep only the connection table.
# This can take a few minutes for a few thousand neurons.
_, out_edges = fetch_adjacencies(sources=neuron_ids, targets=None)
_, in_edges = fetch_adjacencies(sources=None, targets=neuron_ids)

# Note: connection_df has one row per (bodyId_pre, bodyId_post, roi) — the
# same pair of neurons can appear more than once if they connect in multiple
# brain regions. Group by the pair first so weight is a true per-pair total.
out_edges_grouped = out_edges.groupby(["bodyId_pre", "bodyId_post"], as_index=False)[
    "weight"
].sum()
in_edges_grouped = in_edges.groupby(["bodyId_pre", "bodyId_post"], as_index=False)[
    "weight"
].sum()

# ---- Step C: build one feature row per neuron ----
features = []
for bid in neuron_ids:
    out_rows = out_edges_grouped[out_edges_grouped["bodyId_pre"] == bid]
    in_rows = in_edges_grouped[in_edges_grouped["bodyId_post"] == bid]
    features.append(
        {
            "bodyId": bid,
            "out_degree": out_rows["bodyId_post"].nunique(),
            "in_degree": in_rows["bodyId_pre"].nunique(),
            "total_output_weight": out_rows["weight"].sum(),
            "total_input_weight": in_rows["weight"].sum(),
        }
    )

feat_df = pd.DataFrame(features)
feat_df = feat_df.merge(
    neurons[["bodyId", "predictedNt", "type"]], on="bodyId", how="left"
)

feat_df.to_csv("male_cns_central_complex_features.csv", index=False)
print("Saved male_cns_central_complex_features.csv — send this file back to continue.")
