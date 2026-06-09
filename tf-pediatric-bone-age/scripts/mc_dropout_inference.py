#!/usr/bin/env python
"""Monte-Carlo dropout inference + UQ metrics for the saved RSNA bone-age TF models.

Loads a final ``.keras`` model from ``OUTPUT_DIR`` (or ``--model`` flag) and
runs ``--n-samples`` stochastic forward passes per image with dropout left
ON, as in Gal & Ghahramani (2016).

For each requested split (validation, test), saves:
  * ``mc_dropout_<variant>_<split>.csv`` - per-image (true, mean, std, abs_err)
  * a row in ``uq_metrics_<variant>.csv`` with PICP, MPIW, ECE summaries

Usage:
    python scripts/mc_dropout_inference.py --variant baseline --n-samples 50
    python scripts/mc_dropout_inference.py --variant multi --splits test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import uq_metrics  # noqa: E402


_root = Path(__file__).resolve().parent.parent.parent
DEBUG_LOG_PATH = _root / ".cursor/debug-fa1e2c.log"
DEBUG_SESSION_ID = "fa1e2c"


def debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    # Ensure directory exists for debug log
    DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def configure_single_gpu(gpu_index: int = 0) -> None:
    """Restrict TensorFlow runtime to a single GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected. Running inference on CPU.")
        return
    if gpu_index < 0 or gpu_index >= len(gpus):
        raise ValueError(f"--gpu-index must be in [0, {len(gpus) - 1}], got {gpu_index}")

    selected_gpu = gpus[gpu_index]
    tf.config.set_visible_devices(selected_gpu, "GPU")
    tf.config.experimental.set_memory_growth(selected_gpu, True)
    print(f"Using single GPU device: {selected_gpu.name}")


def resolve_paths(output_dir_arg: Path | None = None):
    data_root = Path(
        os.environ.get(
            "BONE_AGE_DATA_ROOT",
            str(_root / "data/rsna_training"),
        )
    ).resolve()

    images_dir = next(
        (
            p
            for p in [
                data_root / "boneage-training-dataset" / "boneage-training-dataset",
                data_root / "boneage-training-dataset",
            ]
            if p.is_dir()
        ),
        data_root / "boneage-training-dataset",
    )
    train_csv = next(
        (p for p in [data_root / "boneage-training-dataset.csv", data_root / "train.csv"] if p.is_file()),
        data_root / "train.csv",
    )

    output_dir = Path(
        output_dir_arg
        or os.environ.get(
            "BONE_AGE_OUTPUT_DIR",
            str(_root / "tf-pediatric-bone-age/outputs/rsna_boneage_models"),
        )
    )
    return data_root, train_csv, images_dir, output_dir


def build_split(train_csv: Path, images_dir: Path, seed: int = 42):
    full_df = pd.read_csv(train_csv)
    if "id" not in full_df.columns and "fileName" in full_df.columns:
        full_df["id"] = full_df["fileName"].astype(str).str.replace(".png", "", regex=False)
    full_df["filepath"] = full_df["id"].astype(str).apply(lambda x: str(images_dir / f"{x}.png"))

    train_df, _tmp = train_test_split(full_df, test_size=0.20, random_state=seed, stratify=full_df["male"])
    val_df, test_df = train_test_split(_tmp, test_size=0.50, random_state=seed, stratify=_tmp["male"])
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    max_age = float(train_df["boneage"].max())
    return train_df, val_df, test_df, max_age


def make_dataset(df: pd.DataFrame, *, multi_input: bool, img_size=(300, 300), batch_size=32):
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

    def parse_single(filepath, age):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = effnet_preprocess(img)
        return img, tf.cast(age, tf.float32)

    def parse_multi(filepath, sex, age):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = effnet_preprocess(img)
        return (
            {"image_input": img, "sex_input": tf.expand_dims(tf.cast(sex, tf.float32), -1)},
            tf.cast(age, tf.float32),
        )

    if multi_input:
        ds = tf.data.Dataset.from_tensor_slices(
            (
                df["filepath"].values,
                df["male"].astype("float32").values,
                df["boneage"].astype("float32").values,
            )
        ).map(parse_multi, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(
            (df["filepath"].values, df["boneage"].astype("float32").values)
        ).map(parse_single, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _tile_batch_tensor(tensor: tf.Tensor, repeats: int) -> tf.Tensor:
    multiples = tf.concat(
        ([repeats], tf.ones(tf.maximum(0, tf.rank(tensor) - 1), dtype=tf.int32)),
        axis=0,
    )
    return tf.tile(tensor, multiples)


def _tile_mc_inputs(inputs, repeats: int):
    if isinstance(inputs, dict):
        return {k: _tile_batch_tensor(v, repeats) for k, v in inputs.items()}
    return _tile_batch_tensor(inputs, repeats)


def mc_predict(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    n_samples: int,
    max_age: float,
    mc_chunk: int = 10,
):
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    truths: List[np.ndarray] = []

    n_batches = int(dataset.cardinality().numpy())
    if n_batches < 0:
        n_batches = None

    for i, batch in enumerate(dataset):
        if isinstance(batch, tuple) and len(batch) == 2:
            inputs, y = batch
        else:
            inputs, y = batch[0], batch[-1]

        if i == 0:
            # region agent log
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="mc_dropout_inference.py:157",
                message="first-batch-shape",
                data={
                    "y_shape": list(y.shape),
                    "n_samples": int(n_samples),
                    "max_age": float(max_age),
                    "input_kind": "dict" if isinstance(inputs, dict) else "tensor",
                    "input_shape": {k: list(v.shape) for k, v in inputs.items()}
                    if isinstance(inputs, dict)
                    else list(inputs.shape),
                },
            )
            # endregion
        try:
            remaining = int(n_samples)
            preds_chunks = []
            batch_n = int(y.shape[0])
            while remaining > 0:
                chunk = min(int(mc_chunk), remaining)
                tiled_inputs = _tile_mc_inputs(inputs, chunk)
                preds_chunk = tf.squeeze(model(tiled_inputs, training=True), axis=-1)
                preds_chunk = tf.reshape(preds_chunk, (chunk, batch_n))
                preds_chunks.append(preds_chunk)
                remaining -= chunk
            preds = tf.concat(preds_chunks, axis=0)
        except Exception as exc:
            # region agent log
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H1",
                location="mc_dropout_inference.py:178",
                message="mc-stack-failed",
                data={"batch_index": int(i), "exception_type": type(exc).__name__, "exception": str(exc)},
            )
            # endregion
            raise
        if i == 0:
            # region agent log
            debug_log(
                run_id="pre-fix",
                hypothesis_id="H1",
                location="mc_dropout_inference.py:189",
                message="first-batch-preds-shape",
                data={"preds_shape": list(preds.shape)},
            )
            # endregion
        preds_months = preds.numpy() * max_age
        means.append(preds_months.mean(axis=0))
        stds.append(preds_months.std(axis=0))
        truths.append(y.numpy())
        if n_batches and (i + 1) % max(1, n_batches // 20) == 0:
            print(f"  batch {i + 1}/{n_batches}")

    return (
        np.concatenate(means).astype(np.float64),
        np.concatenate(stds).astype(np.float64),
        np.concatenate(truths).astype(np.float64),
    )


def write_per_image_csv(path: Path, mu: np.ndarray, sd: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> None:
    out = pd.DataFrame(
        {
            "id": df["id"].astype(str).values,
            "male": df["male"].values,
            "true_boneage": y,
            "pred_mean_boneage": mu,
            "pred_std_boneage": sd,
            "abs_error": np.abs(mu - y),
        }
    )
    out.to_csv(path, index=False)


def write_metrics_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_focused_row(row: dict, alpha: float = 0.95) -> dict:
    """Project the full metrics row down to the requested PICP / MPIW / ECE fields."""

    picp_key = f"picp@{alpha:g}"
    mpiw_key = f"mpiw@{alpha:g}"
    focused = {
        "variant": row.get("variant"),
        "split": row.get("split"),
        "n": row.get("n"),
        "n_samples": row.get("n_samples"),
        "alpha": alpha,
        "PICP": row.get(picp_key),
        "MPIW": row.get(mpiw_key),
        "ECE": row.get("ece_central"),
    }
    return focused


def format_focused_summary(focused: dict) -> str:
    return (
        f"[REPORT] variant={focused['variant']:<8} split={focused['split']:<5} "
        f"n={focused['n']:>5} T={focused['n_samples']:>3} alpha={focused['alpha']:.2f} | "
        f"PICP={focused['PICP']:.4f}  MPIW={focused['MPIW']:.3f}  ECE={focused['ECE']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("baseline", "multi"), default="baseline")
    parser.add_argument("--model", type=Path, default=None, help="Override .keras model path (default: <OUTPUT_DIR>/{variant}_final.keras)")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-index", type=int, default=0, help="Single GPU index to use for inference.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only run on first N rows per split (sanity).")
    parser.add_argument("--splits", nargs="+", choices=("val", "test", "all"), default=["all"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Split seed (must match training).")
    parser.add_argument(
        "--mc-chunk",
        type=int,
        default=10,
        help="Number of MC samples to evaluate in one forward call (speed/memory tradeoff).",
    )
    parser.add_argument(
        "--report-alpha",
        type=float,
        default=0.95,
        help="Confidence level for the focused PICP/MPIW summary (default 0.95).",
    )
    args = parser.parse_args()
    if args.mc_chunk <= 0:
        print("ERROR: --mc-chunk must be >= 1", file=sys.stderr)
        sys.exit(3)

    configure_single_gpu(args.gpu_index)

    _, train_csv, images_dir, output_dir = resolve_paths(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.is_file():
        print(f"ERROR: train CSV not found at {train_csv}", file=sys.stderr)
        sys.exit(1)

    model_default_name = "baseline_final.keras" if args.variant == "baseline" else "multi_input_final.keras"
    model_path = args.model or (output_dir / model_default_name)
    if not model_path.is_file():
        print(
            f"ERROR: final checkpoint for variant '{args.variant}' not found at {model_path}.\n"
            f"  Expected file: {model_default_name}\n"
            f"  Searched directory: {output_dir}\n"
            f"  Hint: train the model first or pass --model <path-to-final.keras>.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    # region agent log
    debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="mc_dropout_inference.py:257",
        message="model-and-gpu-context",
        data={
            "variant": args.variant,
            "model_path": str(model_path),
            "physical_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [d.name for d in tf.config.list_logical_devices("GPU")],
        },
    )
    # endregion

    train_df, val_df, test_df, max_age = build_split(train_csv, images_dir, seed=args.seed)
    if args.limit:
        val_df = val_df.head(args.limit).reset_index(drop=True)
        test_df = test_df.head(args.limit).reset_index(drop=True)
    print(f"Sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)} | max_age={max_age}")

    requested = set(args.splits)
    if "all" in requested:
        requested = {"val", "test"}

    multi = args.variant == "multi"
    split_to_df = {"val": val_df, "test": test_df}
    metrics_rows: List[dict] = []
    focused_rows: List[dict] = []

    for split in ("val", "test"):
        if split not in requested:
            continue
        df = split_to_df[split]
        batch_size = int(args.batch_size)
        mc_chunk = int(args.mc_chunk)
        while True:
            ds = make_dataset(df, multi_input=multi, batch_size=batch_size)
            # region agent log
            debug_log(
                run_id="post-fix",
                hypothesis_id="H5",
                location="mc_dropout_inference.py:285",
                message="split-runtime-config",
                data={
                    "split": split,
                    "rows": int(len(df)),
                    "batch_size": int(batch_size),
                    "n_samples": int(args.n_samples),
                    "mc_chunk": int(mc_chunk),
                    "dataset_cardinality": int(ds.cardinality().numpy()),
                },
            )
            # endregion
            print(
                f"\n[{split}] running {args.n_samples} stochastic forward passes "
                f"(batch_size={batch_size}, mc_chunk={mc_chunk}) ..."
            )
            try:
                mu, sd, y = mc_predict(model, ds, args.n_samples, max_age, mc_chunk=mc_chunk)
                break
            except tf.errors.ResourceExhaustedError:
                # region agent log
                debug_log(
                    run_id="post-fix",
                    hypothesis_id="H5",
                    location="mc_dropout_inference.py:304",
                    message="resource-exhausted-retry",
                    data={"split": split, "failed_batch_size": int(batch_size), "failed_mc_chunk": int(mc_chunk)},
                )
                # endregion
                if mc_chunk > 1:
                    mc_chunk = max(1, mc_chunk // 2)
                    print(f"  OOM detected; retrying split '{split}' with mc_chunk={mc_chunk}")
                    continue
                if batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"  OOM detected; retrying split '{split}' with batch_size={batch_size}")

        per_image_csv = output_dir / f"mc_dropout_{args.variant}_{split}.csv"
        write_per_image_csv(per_image_csv, mu, sd, y, df)
        print(f"  per-image CSV -> {per_image_csv}")

        picp_levels = sorted(set(uq_metrics.DEFAULT_PICP_LEVELS) | {float(args.report_alpha)})
        row = uq_metrics.summarize(
            y,
            mu,
            sd,
            n_samples=args.n_samples,
            split=split,
            picp_levels=picp_levels,
            extra={"variant": args.variant},
        )
        metrics_rows.append(row)
        print(uq_metrics.format_summary(row))

        focused = build_focused_row(row, alpha=float(args.report_alpha))
        focused_rows.append(focused)
        print(format_focused_summary(focused))

    if metrics_rows:
        metrics_csv = output_dir / f"uq_metrics_{args.variant}.csv"
        write_metrics_csv(metrics_csv, metrics_rows)
        print(f"\nSaved UQ metrics summary to {metrics_csv}")

    if focused_rows:
        focused_csv = output_dir / f"uq_report_{args.variant}.csv"
        write_metrics_csv(focused_csv, focused_rows)
        print(f"Saved focused PICP/MPIW/ECE report to {focused_csv}")


if __name__ == "__main__":
    main()
