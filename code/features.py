"""FE dan FR · Ekstraksi, penamaan dan penyusunan 195 fitur."""


import hashlib
import inspect
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import euler_number, label, regionprops
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from tqdm.auto import tqdm


from .acquisition import SEED, read_grayscale
from .enhancement import PREPROCESSING, enhance_image
from .segmentation import segment_image


GLCM_ANGLES_DEG = [0, 45, 90, 135]
pht_seed_argument = (
    "rng" if "rng" in inspect.signature(probabilistic_hough_line).parameters else "seed"
)
PHT_RANDOM_ARGUMENT = {pht_seed_argument: SEED}


def extract_glcm(img_enhanced):
    img_32 = (img_enhanced // 8).astype(np.uint8)

    angles = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
    ]

    glcm = graycomatrix(
        img_32,
        distances=[1],
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True,
    )

    contrast = graycoprops(glcm, "contrast").flatten()
    homogeneity = graycoprops(glcm, "homogeneity").flatten()
    energy = graycoprops(glcm, "energy").flatten()
    correlation = graycoprops(glcm, "correlation").flatten()

    return np.concatenate(
        [
            contrast,
            homogeneity,
            energy,
            correlation,
        ]
    ).astype(np.float32)


def extract_block_lbp(img_enhanced, k=4, P=8, R=1):
    h, w = img_enhanced.shape
    row_edges = np.linspace(0, h, k + 1, dtype=int)
    col_edges = np.linspace(0, w, k + 1, dtype=int)

    lbp_features = []

    lbp_img = local_binary_pattern(
        img_enhanced,
        P,
        R,
        method="uniform",
    )

    for r in range(k):
        for c in range(k):
            block = lbp_img[
                row_edges[r] : row_edges[r + 1],
                col_edges[c] : col_edges[c + 1],
            ]

            hist, _ = np.histogram(
                block.ravel(),
                np.arange(11),
                range=(0, 10),
            )

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-6
            lbp_features.extend(hist)

    return np.asarray(lbp_features, dtype=np.float32)


def extract_pht(mask_final):
    lines = probabilistic_hough_line(
        mask_final,
        threshold=10,
        line_length=15,
        line_gap=5,
        **PHT_RANDOM_ARGUMENT,
    )

    if not lines:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    lengths = []
    angles = []
    vert_len = 0.0
    horiz_len = 0.0

    for line in lines:
        p0, p1 = line
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        length = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(abs(dy), abs(dx)) * (180.0 / np.pi)

        lengths.append(length)
        angles.append(angle)

        if angle >= 45:
            vert_len += length
        else:
            horiz_len += length

    lengths = np.asarray(lengths, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)

    weighted_dominant_angle = np.average(
        angles,
        weights=lengths,
    )

    spatial_ratio = np.log((vert_len + 1.0) / (horiz_len + 1.0))
    num_segments = len(lines)

    return np.array(
        [
            weighted_dominant_angle,
            spatial_ratio,
            num_segments,
        ],
        dtype=np.float32,
    )


def extract_morphology(mask_final):
    """
    16 morphology/topology features.
    """
    mask = (mask_final > 0).astype(np.uint8)
    h, w = mask.shape
    image_area = h * w

    labeled_mask, num_components = label(
        mask,
        connectivity=2,
        return_num=True,
    )

    props = regionprops(labeled_mask)

    if len(props) == 0:
        return np.zeros(16, dtype=np.float32)

    areas = []
    aspect_ratios = []
    eccentricities = []
    solidities = []
    extents = []

    for region in props:
        area = region.area
        areas.append(area)

        major = region.axis_major_length
        minor = region.axis_minor_length

        if minor >= 1.0:
            aspect_ratio = major / minor
        else:
            aspect_ratio = major / 1.0

        aspect_ratios.append(aspect_ratio)
        eccentricities.append(region.eccentricity)
        solidities.append(region.solidity)
        extents.append(region.extent)

    areas = np.asarray(areas, dtype=np.float32)
    aspect_ratios = np.asarray(aspect_ratios, dtype=np.float32)
    eccentricities = np.asarray(eccentricities, dtype=np.float32)
    solidities = np.asarray(solidities, dtype=np.float32)
    extents = np.asarray(extents, dtype=np.float32)

    crack_area = np.sum(mask)
    crack_area_ratio = crack_area / image_area

    mean_area = np.mean(areas)
    std_area = np.std(areas)

    skeleton = skeletonize(mask)
    skeleton_length = np.sum(skeleton)

    kernel = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ]
    )

    filtered = cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        kernel,
    )

    branch_points = np.sum(filtered >= 13)
    endpoint_points = np.sum(filtered == 11)

    branch_density = branch_points / (skeleton_length + 1e-6)
    endpoint_density = endpoint_points / (skeleton_length + 1e-6)

    skeleton_area_ratio = skeleton_length / (crack_area + 1e-6)

    euler_num = euler_number(mask, connectivity=2)
    normalized_euler = euler_num / (skeleton_length + 1e-6)

    return np.array(
        [
            num_components,
            crack_area_ratio,
            mean_area,
            std_area,
            np.mean(aspect_ratios),
            np.std(aspect_ratios),
            np.mean(eccentricities),
            np.std(eccentricities),
            np.mean(solidities),
            np.mean(extents),
            skeleton_length,
            skeleton_area_ratio,
            branch_density,
            endpoint_density,
            euler_num,
            normalized_euler,
        ],
        dtype=np.float32,
    )


def extract_features(path, config):
    _, _, enhanced = enhance_image(read_grayscale(path), config)
    _, _, _, mask = segment_image(enhanced, config)
    vector = np.concatenate(
        (
            extract_glcm(enhanced),
            extract_block_lbp(enhanced),
            extract_pht(mask),
            extract_morphology(mask),
        )
    ).astype(np.float32)
    if vector.shape != (195,) or not np.isfinite(vector).all():
        raise ValueError(f"Vektor fitur tidak valid: {path}")
    return vector


def build_feature_names():
    names = []

    # GLCM: 4 properties × 4 angles
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        for angle in GLCM_ANGLES_DEG:
            names.append(f"glcm_{prop}_{angle}deg")

    # Block-LBP: 4x4 blocks × 10 bins
    for r in range(4):
        for c in range(4):
            for b in range(10):
                names.append(f"blbp_r{r}_c{c}_bin{b}")

    # PHT: 3
    names.extend(
        [
            "pht_weighted_dominant_angle",
            "pht_vertical_horizontal_ratio",
            "pht_num_segments",
        ]
    )

    # Morphology/topology: 16
    names.extend(
        [
            "morph_num_components",
            "morph_crack_area_ratio",
            "morph_mean_component_area",
            "morph_std_component_area",
            "morph_mean_aspect_ratio",
            "morph_std_aspect_ratio",
            "morph_mean_eccentricity",
            "morph_std_eccentricity",
            "morph_mean_solidity",
            "morph_mean_extent",
            "morph_skeleton_length",
            "morph_skeleton_area_ratio",
            "morph_branch_density",
            "morph_endpoint_density",
            "morph_euler_number",
            "morph_normalized_euler",
        ]
    )

    return names


FEATURE_NAMES = build_feature_names()
FEATURE_GROUPS = {
    "GLCM": np.arange(0, 16),
    "Block-LBP": np.arange(16, 176),
    "PHT": np.arange(176, 179),
    "Morphology": np.arange(179, 195),
}


def write_feature_names(out):
    pd.DataFrame({"column": range(195), "feature": FEATURE_NAMES}).to_csv(
        out.tables / "feature_names.csv", index=False
    )


LEGACY_CACHE_DIRS = [Path("/kaggle/working/crack_reproducible/cache")]
LEGACY_CODE_HASH = "0f9407b6239b955b80c58d345034f3a1d30e2f333716215d238c3898b6fa24ca"

FEATURE_FUNCTIONS = (
    read_grayscale,
    enhance_image,
    segment_image,
    extract_features,
    extract_glcm,
    extract_block_lbp,
    extract_pht,
    extract_morphology,
)
CODE_HASH = hashlib.sha256(
    "".join(inspect.getsource(function) for function in FEATURE_FUNCTIONS).encode()
).hexdigest()


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def read_legacy_cache(split, rows, raw_manifest, env):
    raw_rows = raw_manifest.loc[raw_manifest["split"].eq(split)]
    raw_hash = hashlib.sha256("".join(raw_manifest["sha256"]).encode()).hexdigest()
    identity = {
        "cfg": PREPROCESSING,
        "split": split,
        "data": raw_hash,
        "code": LEGACY_CODE_HASH,
        "env": env,
        "seed": SEED,
    }
    positions = raw_rows.index.get_indexer(rows.index)
    if np.any(positions < 0):
        raise RuntimeError("Baris manifest bersih tidak ditemukan dalam manifest awal.")
    filename = f"{cache_key(identity)}.npz"
    for directory in LEGACY_CACHE_DIRS:
        candidate = directory / filename
        if not candidate.is_file():
            continue
        try:
            with np.load(candidate, allow_pickle=False) as cache:
                matrix = cache["X"]
            if (
                matrix.shape == (len(raw_rows), len(FEATURE_NAMES))
                and np.isfinite(matrix).all()
            ):
                print(
                    f"[{split}] Cache lama cocok, memakai {len(rows):,}/{len(raw_rows):,} baris.",
                    flush=True,
                )
                return matrix[positions]
        except (OSError, ValueError, KeyError) as exc:
            print(f"[{split}] Cache lama tidak dapat dibaca: {exc}", flush=True)
    return None


def feature_matrix(split, manifest, out, env, raw_manifest):
    rows = manifest.loc[manifest["split"].eq(split)]
    paths = rows["path"].to_numpy(dtype=str)
    hashes = rows["sha256"].to_numpy(dtype=str)
    identity = {
        "split": split,
        "files": list(zip(paths.tolist(), hashes.tolist())),
        "cfg": PREPROCESSING,
        "code": CODE_HASH,
        "env": env,
        "seed": SEED,
        "features": FEATURE_NAMES,
    }
    cache_path = out.cache / f"{split}_{cache_key(identity)}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            matrix = cache["X"]
            if not np.array_equal(cache["paths"], paths):
                raise ValueError(f"Urutan path cache tidak cocok: {cache_path}")
        print(f"[{split}] Memuat cache fitur bersih.", flush=True)
    else:
        matrix = read_legacy_cache(split, rows, raw_manifest, env)
        if matrix is None:
            print(f"[{split}] Ekstraksi {len(rows):,} gambar.", flush=True)
            matrix = np.stack(
                [
                    extract_features(path, PREPROCESSING)
                    for path in tqdm(paths, desc=f"Ekstraksi {split}")
                ]
            )
        with NamedTemporaryFile(dir=out.cache, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary_path, X=matrix, paths=paths)
            os.replace(temporary_path, cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    if matrix.shape != (len(rows), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise ValueError(f"Matriks fitur {split} tidak valid: {matrix.shape}")
    print(f"[{split}] {matrix.shape[0]:,} gambar × {matrix.shape[1]} fitur.", flush=True)
    return matrix, rows.reset_index(drop=True)
