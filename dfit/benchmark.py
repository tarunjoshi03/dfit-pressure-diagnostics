"""
DFITBench: a standardized synthetic benchmark for DFIT interpretation methods.

The central problem in DFIT research is the absence of a public dataset with
known ground truth. Every ML paper on DFIT closes with the same limitation:
"real DFIT data is proprietary." This module generates a standardized benchmark
that any researcher can reproduce exactly - covering all four leakoff regimes,
a range of formation types, noise levels, and pump times - so that different
interpretation methods can be compared on a common footing.

Design
------
The benchmark is parameterized across three axes:

1. Leakoff regime (4 classes): normal, pressure_dependent, height_recession,
   tip_extension.

2. Formation type (3 classes), which controls the physical parameters:
   - tight_gas   : moderate depth, lower ISIP, shorter closure time
   - shale       : deeper, higher ISIP, longer closure time (lower perm)
   - conventional: shallower, higher permeability, faster closure

3. Noise level (3 levels):
   - low    : 1.0 psi - lab-quality or high-resolution memory gauge
   - medium : 3.5 psi - typical field memory gauge
   - high   : 8.0 psi - surface gauge or noisy wellbore condition

Every combination (4 × 3 × 3 = 36 cells) is populated with n_per_cell tests,
using random seeds for reproducibility. Default n_per_cell = 28 gives 1,008
tests total.

Each record in the benchmark carries the full ground truth alongside the raw
time/pressure/rate arrays, so a researcher can evaluate any interpretation
method without re-running the generator.

References
----------
Mohamed et al. (2020) URTeC-2020-2762 (noted DFIT data rarity).
Barree, Miskimins, Gilbert (2015) SPE-169539-PA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .synthetic import generate_dfit, SyntheticDFIT

# ---- Formation parameter presets -------------------------------------------

FORMATION_PRESETS = {
    "tight_gas": dict(
        isip_psi=7200,
        closure_pressure_psi=6100,
        closure_G=7.0,
        reservoir_pressure_psi=4800,
        t_pump_min=5.0,
        G_max=28.0,
        n_points=1200,
    ),
    "shale": dict(
        isip_psi=9400,
        closure_pressure_psi=7900,
        closure_G=10.0,
        reservoir_pressure_psi=6200,
        t_pump_min=8.0,
        G_max=40.0,
        n_points=1400,
    ),
    "conventional": dict(
        isip_psi=5400,
        closure_pressure_psi=4600,
        closure_G=4.5,
        reservoir_pressure_psi=3200,
        t_pump_min=3.0,
        G_max=16.0,
        n_points=900,
    ),
}

NOISE_LEVELS = {
    "low": 1.0,
    "medium": 3.5,
    "high": 8.0,
}

REGIMES = ("normal", "pressure_dependent", "height_recession", "tip_extension")
FORMATIONS = ("tight_gas", "shale", "conventional")
NOISE_LABELS = ("low", "medium", "high")


# ---- Benchmark record -------------------------------------------------------

@dataclass
class BenchmarkRecord:
    """One labeled DFIT in the benchmark."""
    record_id: int
    regime: str
    formation: str
    noise_label: str
    noise_psi: float
    seed: int
    # ground truth
    isip_psi: float
    closure_pressure_psi: float
    closure_G: float
    closure_time_min: float
    net_pressure_psi: float
    reservoir_pressure_psi: float
    t_pump_min: float
    # raw data (stored as lists for JSON serialisation)
    time_min: list
    pressure_psi: list
    rate_bpm: list
    G: list


# ---- Generator --------------------------------------------------------------

def generate_benchmark(
    n_per_cell: int = 28,
    seed_offset: int = 0,
    regimes: Sequence[str] = REGIMES,
    formations: Sequence[str] = FORMATIONS,
    noise_labels: Sequence[str] = NOISE_LABELS,
    verbose: bool = True,
) -> list[BenchmarkRecord]:
    """
    Generate the full DFITBench dataset.

    Parameters
    ----------
    n_per_cell : int
        Number of tests per (regime × formation × noise) combination.
        Default 28 gives 4×3×3×28 = 1,008 total records.
    seed_offset : int
        Added to every seed for reproducibility across benchmark versions.
    regimes, formations, noise_labels : sequences
        Subsets to generate (default: all).
    verbose : bool
        Print progress.

    Returns
    -------
    list of BenchmarkRecord
    """
    records = []
    record_id = 0

    total = len(regimes) * len(formations) * len(noise_labels) * n_per_cell
    done = 0

    for regime in regimes:
        for formation in formations:
            params = FORMATION_PRESETS[formation].copy()
            for noise_label in noise_labels:
                noise_psi = NOISE_LEVELS[noise_label]
                for i in range(n_per_cell):
                    seed = seed_offset + record_id
                    d = generate_dfit(
                        regime=regime,
                        noise_psi=noise_psi,
                        seed=seed,
                        **params,
                    )
                    rec = BenchmarkRecord(
                        record_id=record_id,
                        regime=regime,
                        formation=formation,
                        noise_label=noise_label,
                        noise_psi=noise_psi,
                        seed=seed,
                        isip_psi=d.truth["isip_psi"],
                        closure_pressure_psi=d.truth["closure_pressure_psi"],
                        closure_G=d.truth["closure_G"],
                        closure_time_min=d.truth["closure_time_min"],
                        net_pressure_psi=d.truth["net_pressure_psi"],
                        reservoir_pressure_psi=d.truth["reservoir_pressure_psi"],
                        t_pump_min=d.truth["t_pump_min"],
                        time_min=d.time_min.tolist(),
                        pressure_psi=d.pressure_psi.tolist(),
                        rate_bpm=d.rate_bpm.tolist(),
                        G=[float("nan") if np.isnan(g) else g
                           for g in d.G.tolist()],
                    )
                    records.append(rec)
                    record_id += 1
                    done += 1

                if verbose:
                    pct = done / total * 100
                    print(
                        f"  {regime:20s} | {formation:12s} | {noise_label:6s} "
                        f"| {n_per_cell} records  [{pct:5.1f}%]"
                    )

    if verbose:
        print(f"\nDFITBench: {len(records)} records generated.")
    return records


def benchmark_to_dataframe(records: list[BenchmarkRecord]) -> pd.DataFrame:
    """
    Convert benchmark records to a metadata DataFrame (no raw arrays).
    Useful for stratified sampling and result aggregation.
    """
    rows = []
    for r in records:
        rows.append({
            "record_id": r.record_id,
            "regime": r.regime,
            "formation": r.formation,
            "noise_label": r.noise_label,
            "noise_psi": r.noise_psi,
            "seed": r.seed,
            "isip_psi": r.isip_psi,
            "closure_pressure_psi": r.closure_pressure_psi,
            "closure_G": r.closure_G,
            "closure_time_min": r.closure_time_min,
            "net_pressure_psi": r.net_pressure_psi,
            "reservoir_pressure_psi": r.reservoir_pressure_psi,
            "t_pump_min": r.t_pump_min,
        })
    return pd.DataFrame(rows)


def save_benchmark(
    records: list[BenchmarkRecord],
    output_dir: str | Path = "data/benchmark",
) -> None:
    """
    Save the benchmark to disk.

    Writes:
    - data/benchmark/metadata.csv  (labels + ground truth, no raw arrays)
    - data/benchmark/records.jsonl (one JSON record per line, includes arrays)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # metadata CSV
    meta = benchmark_to_dataframe(records)
    meta.to_csv(output_dir / "metadata.csv", index=False)

    # full records as JSONL (one per line, streamable)
    with open(output_dir / "records.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")

    print(f"Saved {len(records)} records to {output_dir}/")
    print(f"  metadata.csv : {(output_dir/'metadata.csv').stat().st_size/1024:.0f} KB")
    print(f"  records.jsonl: {(output_dir/'records.jsonl').stat().st_size/1024:.0f} KB")


def load_benchmark(
    input_dir: str | Path = "data/benchmark",
) -> list[BenchmarkRecord]:
    """Load benchmark records from disk."""
    input_dir = Path(input_dir)
    records = []
    with open(input_dir / "records.jsonl") as f:
        for line in f:
            d = json.loads(line)
            d["time_min"] = np.array(d["time_min"])
            d["pressure_psi"] = np.array(d["pressure_psi"])
            d["rate_bpm"] = np.array(d["rate_bpm"])
            d["G"] = np.array(d["G"])
            records.append(BenchmarkRecord(**d))
    return records
