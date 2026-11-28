"""S · Otsu, Canny, fusi spasial dan seleksi komponen."""


import cv2
import numpy as np
import matplotlib.pyplot as plt


from .acquisition import SOURCE_LABELS, read_grayscale
from .enhancement import PREPROCESSING, enhance_image


def segment_image(enhanced, config):
    threshold, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny = cv2.Canny(
        enhanced, config["canny_low"] * threshold, config["canny_high"] * threshold
    )
    kernel = np.ones((config["dilate"],) * 2, dtype=np.uint8)
    fused = cv2.bitwise_and(canny, cv2.dilate(otsu, kernel))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(fused, connectivity=8)
    keep_component = stats[:, cv2.CC_STAT_AREA] >= config["area"]
    keep_component[0] = False
    mask = keep_component[labels].astype(np.uint8)
    return otsu, canny, fused, mask


def plot_pipeline(out, manifest):
    samples = (
        manifest.loc[manifest["split"].eq("train")]
        .sort_values(["label_id", "path"])
        .groupby(["source", "label"], sort=False)
        .head(1)
    )
    titles = (
        "Grayscale",
        "Gaussian",
        "CLAHE",
        "Black-hat",
        "Otsu",
        "Canny",
        "Irisan",
        "Mask akhir",
    )
    for source, rows in samples.groupby("source", sort=False):
        fig, axes = plt.subplots(
            len(rows),
            len(titles),
            figsize=(16, 2.6 * len(rows)),
            squeeze=False,
            layout="constrained",
        )
        for row, panel in zip(rows.itertuples(index=False), axes):
            raw = read_grayscale(row.path)
            blurred, equalized, enhanced = enhance_image(raw, PREPROCESSING)
            otsu, canny, fused, mask = segment_image(enhanced, PREPROCESSING)
            for ax, image, title in zip(
                panel,
                (raw, blurred, equalized, enhanced, otsu, canny, fused, mask * 255),
                titles,
            ):
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{row.label}\n{title}", fontsize=9)
                ax.axis("off")
        fig.suptitle(f"Tahapan pengolahan citra — {SOURCE_LABELS[source]}")
        out.save_figure(fig, f"04_pipeline_{source}.png")
