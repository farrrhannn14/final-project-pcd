"""A · Memuat citra, mengaudit duplikat dan mendeskripsikan split."""


import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from PIL import Image
from tqdm.auto import tqdm


SEED = 42
CPU_THREADS = 1
CLASS_NAMES = ["alligator", "longitudinal", "transverse"]
CLASS_IDS = list(range(len(CLASS_NAMES)))
LABEL_TO_ID = dict(zip(CLASS_NAMES, CLASS_IDS))
SPLITS = ("train", "val", "test")

DATASETS = {
    "pavebench": {
        "root": Path(
            "/kaggle/input/datasets/ahmadfarhanh/"
            "pavebench-dataset/pavebench/data/Distress_Classification"
        ),
        "split_folders": {"train": "train", "val": "val", "test": "test"},
        "class_folders": dict(
            zip(
                CLASS_NAMES,
                (
                    "alligator_crack",
                    "longitudinal_crack",
                    "transverse_crack",
                ),
            )
        ),
    },
    "crack-dataset": {
        "root": Path(
            "/kaggle/input/datasets/ahmadfarhanh/"
            "crack-dataset/pavement-cracks-classification"
        ),
        "split_folders": {"train": "train", "val": "valid", "test": "test"},
        "class_folders": dict(
            zip(
                CLASS_NAMES,
                (
                    "alligator cracking",
                    "longitudinal cracking",
                    "transverse cracking",
                ),
            )
        ),
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOURCE_LABELS = {"pavebench": "PaveBench", "crack-dataset": "Crack Dataset"}
MODEL_COLORS = {"SVM": "#285c9f", "XGBoost": "#d07727"}
CLASS_COLORS = ["#285c9f", "#29917c", "#d07727"]


# Path bawaan sesuai mount pada notebook Kaggle sumber.
for source, variable in (("pavebench", "PAVEBENCH_ROOT"), ("crack-dataset", "CRACK_DATASET_ROOT")):
    if os.environ.get(variable):
        DATASETS[source]["root"] = Path(os.environ[variable])


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    tables: Path
    models: Path
    cache: Path

    def save_figure(self, fig, filename, *, show=True):
        fig.savefig(self.figures / filename, dpi=220, bbox_inches="tight", facecolor="white")
        if show:
            plt.show()
        plt.close(fig)


def prepare_output(root=None):
    """Siapkan direktori dan catat versi runtime untuk satu eksekusi."""
    root = Path(root or os.environ.get("PCD_OUTPUT_DIR", "/kaggle/working/pcd_aesferm25_clip1"))
    out = OutputPaths(root, root / "figures", root / "tables", root / "models", root / "cache")
    for directory in (out.figures, out.tables, out.models, out.cache):
        directory.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(CPU_THREADS)
    cv2.ocl.setUseOpenCL(False)
    packages = ("numpy", "scipy", "pandas", "scikit-learn", "xgboost", "joblib", "scikit-image")
    env = {name: version(name) for name in packages}
    env.update(opencv=cv2.__version__, python=platform.python_version())
    (out.root / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    print(f"Output: {out.root} | CPU | Seed: {SEED}", flush=True)
    return out, env


def read_grayscale(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or min(image.shape) < 4:
        raise ValueError(f"Gambar gagal dibaca atau terlalu kecil untuk Block-LBP: {path}")
    return image


def discover_images(datasets):
    records = []
    for source, spec in datasets.items():
        for split, split_folder in spec["split_folders"].items():
            for class_name, class_folder in spec["class_folders"].items():
                folder = spec["root"] / split_folder / class_folder
                if not folder.is_dir():
                    raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")
                paths = sorted(
                    path
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                if not paths:
                    raise ValueError(f"Folder kelas tidak berisi gambar: {folder}")
                for path in paths:
                    records.append(
                        {
                            "path": str(path),
                            "source": source,
                            "split": split,
                            "label": class_name,
                            "label_id": LABEL_TO_ID[class_name],
                        }
                    )
    return (
        pd.DataFrame(records)
        .sort_values(["split", "source", "label", "path"], kind="stable")
        .reset_index(drop=True)
    )


def exclude_duplicates(manifest):
    audit = manifest.copy()
    stats = audit.groupby("sha256", sort=False).agg(
        n_files=("path", "size"),
        n_splits=("split", "nunique"),
        n_labels=("label", "nunique"),
    )
    audit = audit.join(stats, on="sha256")
    audit["decision"] = "retained"
    audit.loc[audit["n_labels"].gt(1), "decision"] = "label_conflict"

    priority = {"test": 0, "val": 1, "train": 2}
    eligible = audit.loc[audit["decision"].eq("retained")].copy()
    eligible["priority"] = eligible["split"].map(priority)
    ordered = eligible.sort_values(["priority", "source", "path"], kind="stable")
    extra = ordered.index[ordered.duplicated("sha256", keep="first")]
    audit.loc[extra, "decision"] = "extra_copy"

    kept = audit.loc[audit["decision"].eq("retained"), manifest.columns].copy()
    if kept["sha256"].duplicated().any():
        raise RuntimeError("Masih ada hash identik pada manifest bersih.")
    return kept, audit


def acquire(out):
    raw_manifest = discover_images(DATASETS)
    raw_manifest["sha256"] = [
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in tqdm(raw_manifest["path"], desc="Audit SHA-256")
    ]
    manifest, duplicate_audit = exclude_duplicates(raw_manifest)
    excluded = duplicate_audit.loc[duplicate_audit["decision"].ne("retained")].copy()

    expected_cells = pd.MultiIndex.from_product(
        [list(DATASETS), SPLITS, CLASS_NAMES], names=["source", "split", "label"]
    )
    counts = (
        pd.DataFrame(
            {
                "before": raw_manifest.groupby(["source", "split", "label"]).size(),
                "retained": manifest.groupby(["source", "split", "label"]).size(),
            }
        )
        .reindex(expected_cells)
        .fillna(0)
        .astype(int)
    )
    counts["excluded"] = counts["before"] - counts["retained"]
    if counts["retained"].eq(0).any():
        empty = counts.index[counts["retained"].eq(0)].tolist()
        raise ValueError(f"Kombinasi sumber, split, dan kelas kosong setelah eksklusi: {empty}")

    raw_manifest.to_csv(out.tables / "manifest_original.csv", index=False)
    manifest.to_csv(out.tables / "manifest_clean.csv", index=False)
    duplicate_audit.loc[duplicate_audit["n_files"].gt(1)].to_csv(
        out.tables / "duplicate_audit.csv", index=False
    )
    excluded.to_csv(out.tables / "duplicate_exclusions.csv", index=False)
    counts.to_csv(out.tables / "dataset_counts.csv")
    print(
        f"Gambar ditemukan: {len(raw_manifest):,} | Dipakai: {len(manifest):,} | Eksklusi: {len(excluded):,}"
    )
    display(counts)

    return raw_manifest, manifest, counts, excluded


def image_dimensions(path):
    with Image.open(path) as image:
        return image.size


def plot_acquisition(out, manifest, counts, excluded):
    fig, axes = plt.subplots(
        1, len(DATASETS), figsize=(11, 4.2), sharey=True, layout="constrained"
    )
    for ax, source in zip(axes, DATASETS):
        panel = counts.loc[source, "retained"].unstack("label").reindex(SPLITS)
        panel[CLASS_NAMES].plot.bar(ax=ax, color=CLASS_COLORS, rot=0)
        ax.set(title=SOURCE_LABELS[source], xlabel="Split", ylabel="Jumlah gambar")
        ax.legend(title="Kelas", fontsize=8)
    fig.suptitle("Distribusi kelas setelah penyaringan")
    out.save_figure(fig, "01_dataset_distribution.png")

    removed_by_reason = (
        excluded.groupby(["split", "decision"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLITS, columns=["label_conflict", "extra_copy"], fill_value=0)
    )
    removed_by_reason.columns = ["Konflik label", "Salinan berlebih"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    removed_by_reason.plot.bar(stacked=True, color=["#a34950", "#647e9a"], rot=0, ax=ax)
    ax.set(title="Eksklusi file identik per split", xlabel="Split", ylabel="Jumlah gambar")
    out.save_figure(fig, "02_duplicate_exclusions.png")


    resolution = manifest.loc[manifest["split"].isin(["train", "val"])].copy()
    resolution[["width", "height"]] = [
        image_dimensions(path) for path in tqdm(resolution["path"], desc="Audit dimensi")
    ]
    resolution["aspect_ratio"] = resolution["width"] / resolution["height"]
    resolution.to_csv(out.tables / "resolution_audit.csv", index=False)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(10.5, 4), layout="constrained")
    for ax, source in zip(axes, DATASETS):
        rows = resolution.loc[resolution["source"].eq(source)]
        ax.scatter(rows["width"], rows["height"], s=12, alpha=0.25, color="#285c9f")
        ax.set(title=SOURCE_LABELS[source], xlabel="Lebar (piksel)", ylabel="Tinggi (piksel)")
    fig.suptitle("Resolusi gambar train dan validasi")
    out.save_figure(fig, "03_image_resolution.png")
