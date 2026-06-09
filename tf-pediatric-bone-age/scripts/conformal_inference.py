#!/usr/bin/env python
"""Split-conformal inference + PICP/MPIW metrics for saved RSNA bone-age TF models.

This script loads an existing Keras model and performs deterministic inference.
It calibrates split-conformal residuals on the validation split and evaluates
prediction intervals on requested split(s).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
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


_root = Path(__file__).resolve().parent.parent.parent

def resolve_paths(output_dir_arg: Path | None = None) -> tuple[Path, Path, Path, Path]:
    """Resolve data and output paths from arguments/environment."""

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


def build_split(train_csv: Path, images_dir: Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Rebuild the train/val/test split used by training/evaluation scripts."""

    full_df = pd.read_csv(train_csv)
    if "id" not in full_df.columns and "fileName" in full_df.columns:
        full_df["id"] = full_df["fileName"].astype(str).str.replace(".png", "", regex=False)
    full_df["filepath"] = full_df["id"].astype(str).apply(lambda x: str(images_dir / f"{x}.png"))

    train_df, tmp_df = train_test_split(full_df, test_size=0.20, random_state=seed, stratify=full_df["male"])
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=seed, stratify=tmp_df["male"])
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    max_age = float(train_df["boneage"].max())
    
    return train_df, val_df, test_df, max_age


def make_dataset(df: pd.DataFrame, *, multi_input: bool, img_size: tuple[int, int] = (300, 300), batch_size: int = 32) -> tf.data.Dataset:
    """Build inference dataset for baseline or multi-input model."""

    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

    def parse_single(filepath: tf.Tensor, age: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = effnet_preprocess(img)
        return img, tf.cast(age, tf.float32)

    def parse_multi(filepath: tf.Tensor, sex: tf.Tensor, age: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
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


def deterministic_predict(model: tf.keras.Model, dataset: tf.data.Dataset, max_age: float) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic model inference and return predictions and labels."""

    preds: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    for batch in dataset:
        if isinstance(batch, tuple) and len(batch) == 2:
            inputs, y = batch
        else:
            inputs, y = batch[0], batch[-1]

        pred = tf.squeeze(model(inputs, training=False), axis=-1).numpy().astype(np.float64)
        pred = pred * max_age
        preds.append(pred)
        truths.append(y.numpy().astype(np.float64))
    return np.concatenate(preds), np.concatenate(truths)


def resolve_model_path(output_dir: Path, variant: str, override: Path | None) -> Path:
    """Select model path preferring best checkpoint then final model."""

    if override is not None:
        return override

    if variant == "baseline":
        best_candidates = sorted(output_dir.glob("best_baseline_*.keras"))
        fallback = output_dir / "baseline_final.keras"
    else:
        best_candidates = sorted(output_dir.glob("best_multi_input_*.keras"))
        fallback = output_dir / "multi_input_final.keras"

    if best_candidates:
        return best_candidates[-1]
    return fallback


def write_metrics_csv(path: Path, rows: Iterable[dict]) -> None:
    """Write arbitrary rows to CSV preserving first-seen key order."""

    rows = list(rows)
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_levels(levels_arg: str) -> List[float]:
    """Parse comma-separated confidence levels."""

    levels: List[float] = []
    for token in levels_arg.split(","):
        token = token.strip()
        if not token:
            continue
        level = float(token)
        if not 0.0 < level < 1.0:
            raise ValueError(f"Confidence level must be in (0, 1), got {level}")
        levels.append(level)
    if not levels:
        raise ValueError("At least one confidence level is required")
    return sorted(set(levels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "multi"), default="baseline")
    parser.add_argument("--model", type=Path, default=None, help="Override model path")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-index", type=int, default=0, help="Single GPU index to use for inference.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only run on first N rows per split.")
    parser.add_argument("--splits", nargs="+", choices=("val", "test", "all"), default=["test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument(
        "--levels",
        type=str,
        default="0.90,0.95",
        help="Comma-separated confidence levels for conformal intervals, e.g. '0.9,0.95'",
    )
    args = parser.parse_args()

    configure_single_gpu(args.gpu_index)
    levels = parse_levels(args.levels)

    _, train_csv, images_dir, output_dir = resolve_paths(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.is_file():
        print(f"ERROR: train CSV not found at {train_csv}", file=sys.stderr)
        sys.exit(1)

    model_path = resolve_model_path(output_dir, args.variant, args.model)
    if not model_path.is_file():
        print(
            f"ERROR: model for variant '{args.variant}' not found at {model_path}.\n"
            "  Hint: train/export model first or pass --model <path-to-model.keras>.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)

    train_df, val_df, test_df, max_age = build_split(train_csv, images_dir, seed=args.seed)
    if args.limit:
        val_df = val_df.head(args.limit).reset_index(drop=True)
        test_df = test_df.head(args.limit).reset_index(drop=True)

    requested = set(args.splits)
    if "all" in requested:
        requested = {"val", "test"}

    multi_input = args.variant == "multi"
    cal_ds = make_dataset(val_df, multi_input=multi_input, batch_size=int(args.batch_size))
    y_cal_pred, y_cal_true = deterministic_predict(model, cal_ds, max_age)
    cal_residuals = np.abs(y_cal_true - y_cal_pred)
    print(f"Calibration rows={len(y_cal_true)}")

    metrics_rows: List[dict] = []
    report_rows: List[dict] = []

    split_to_df = {"val": val_df, "test": test_df}
    for split in ("val", "test"):
        if split not in requested:
            continue
        df = split_to_df[split]
        ds = make_dataset(df, multi_input=multi_input, batch_size=int(args.batch_size))
        y_pred, y_true = deterministic_predict(model, ds, max_age)
        abs_err = np.abs(y_pred - y_true)

        per_image = {
            "id": df["id"].astype(str).values,
            "male": df["male"].values,
            "true_boneage": y_true,
            "pred_boneage": y_pred,
            "abs_error": abs_err,
        }

        split_metrics_base = uq_metrics.regression_metrics(y_true, y_pred)
        split_rows: List[dict] = []
        for alpha in levels:
            q_hat = uq_metrics.conformal_residual_quantile(cal_residuals, alpha=alpha)
            lower, upper = uq_metrics.conformal_interval(y_pred, q_hat=q_hat)
            coverage = ((y_true >= lower) & (y_true <= upper)).astype(np.int32)

            alpha_key = f"{alpha:g}"
            per_image[f"lower@{alpha_key}"] = lower
            per_image[f"upper@{alpha_key}"] = upper
            per_image[f"covered@{alpha_key}"] = coverage

            row = {
                "variant": args.variant,
                "split": split,
                "model_path": str(model_path),
                "n": int(len(y_true)),
                "n_cal": int(len(y_cal_true)),
                "alpha": float(alpha),
                "q_hat": float(q_hat),
                "PICP": uq_metrics.picp_interval(y_true, lower, upper),
                "MPIW": uq_metrics.mpiw_interval(lower, upper),
                "mae": split_metrics_base["mae"],
                "rmse": split_metrics_base["rmse"],
            }
            split_rows.append(row)
            print(
                f"[{split}] alpha={alpha:.2f} q_hat={q_hat:.3f} "
                f"PICP={row['PICP']:.4f} MPIW={row['MPIW']:.3f}"
            )

        per_image_df = pd.DataFrame(per_image)
        per_image_csv = output_dir / f"conformal_{args.variant}_{split}.csv"
        per_image_df.to_csv(per_image_csv, index=False)
        print(f"  per-image CSV -> {per_image_csv}")

        metrics_rows.extend(split_rows)
        report_rows.extend(
            {
                "variant": row["variant"],
                "split": row["split"],
                "alpha": row["alpha"],
                "PICP": row["PICP"],
                "MPIW": row["MPIW"],
            }
            for row in split_rows
        )

    metrics_csv = output_dir / f"conformal_metrics_{args.variant}.csv"
    report_csv = output_dir / f"conformal_report_{args.variant}.csv"
    write_metrics_csv(metrics_csv, metrics_rows)
    write_metrics_csv(report_csv, report_rows)
    print(f"\nSaved conformal metrics summary to {metrics_csv}")
    print(f"Saved focused conformal report to {report_csv}")


if __name__ == "__main__":
    main()
