"""Train-only pretrained ResNet18 embeddings and a video-diverse review queue."""

import json
import cv2
import numpy as np
import pandas as pd
from .common import ROOT, DATA, REPORTS, setup, save_json


def run():
    setup()
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    from PIL import Image
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.ensemble import IsolationForest
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_num_threads(2)
    torch.hub.set_dir(str(ROOT / ".cache" / "torch"))
    records = [r for r in json.loads((DATA / "manifest.json").read_text()) if r["split"] == "train"]
    weights = ResNet18_Weights.IMAGENET1K_V1
    network = resnet18(weights=weights).eval()
    network.fc = torch.nn.Identity()
    transform = weights.transforms()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(records), 32):
            images = []
            for r in records[start : start + 32]:
                with Image.open(DATA / "images" / "train" / r["file_name"]) as im:
                    images.append(transform(im.convert("RGB")))
            batches.append(network(torch.stack(images)).numpy())
    embeddings = np.concatenate(batches)
    np.save(DATA / "resnet18_embeddings.npy", embeddings)
    standardized = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=32, random_state=42)
    reduced = pca.fit_transform(standardized)
    clusters = KMeans(n_clusters=8, n_init=10, random_state=42).fit_predict(reduced)
    scores = -IsolationForest(n_estimators=250, random_state=42).fit(reduced).score_samples(reduced)
    df = pd.DataFrame(
        [
            {
                "file_name": r["file_name"],
                "video": r["group"],
                "cluster": int(clusters[i]),
                "review_score": float(scores[i]),
                "pca_x": float(reduced[i, 0]),
                "pca_y": float(reduced[i, 1]),
            }
            for i, r in enumerate(records)
        ]
    ).sort_values("review_score", ascending=False)
    df.to_csv(REPORTS / "embedding_review.csv", index=False)
    diverse = df.drop_duplicates("video").head(12)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax, (_, r) in zip(axes.flat, diverse.iterrows()):
        image = cv2.imread(str(DATA / "images" / "train" / r["file_name"]))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{r['video']} / score {r['review_score']:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "embedding_contact_sheet.jpg", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, s=12, alpha=0.65, cmap="tab10")
    ax.set(title="Training images: pretrained visual embeddings", xlabel="PCA 1", ylabel="PCA 2")
    fig.tight_layout()
    fig.savefig(REPORTS / "embedding_clusters.png", dpi=150)
    plt.close(fig)
    save_json(
        REPORTS / "embeddings.json",
        {
            "extractor": "torchvision ResNet18 IMAGENET1K_V1, frozen, 512-D average-pool features",
            "fit_split": "train only",
            "images": len(records),
            "pca_components": 32,
            "explained_variance": float(pca.explained_variance_ratio_.sum()),
            "pipeline": "StandardScaler -> PCA(32) -> KMeans(8), IsolationForest(250)",
            "review_policy": "At most one image per source video in the top-12 contact sheet",
            "interpretation": "Pretrained visual features capture appearance/content, but clusters have no verified semantic labels. No data automatically removed.",
            "top_review_candidates": diverse.to_dict(orient="records"),
        },
    )


if __name__ == "__main__":
    run()
