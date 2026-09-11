'''Probe 1 — Category probe: Can a linear classifier predict the object category from the CLIP embedding? If yes, CLIP encodes shape/identity information.

Probe 2 — Lighting probe: Can a linear classifier predict the lighting condition from the CLIP embedding? If yes, CLIP encodes color information — that's the shortcut.
'''


import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import clip
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RENDERS_DIR = 'renders'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LIGHTING_CONDITIONS = ['neutral', 'warm', 'cool', 'green']

def load_renders(renders_dir):
    embeddings = []
    category_labels = []
    lighting_labels = []

    model, preprocess = clip.load('ViT-B/32', device=DEVICE)
    model.eval()

    files = sorted(os.listdir(renders_dir))
    print(f"Loading {len(files)} renders...")

    with torch.no_grad():
        for fname in files:
            if not fname.endswith('.png'):
                continue

            # filename format: category__stem__lighting.png
            parts = fname.replace('.png', '').split('__')
            if len(parts) != 3:
                continue

            category, stem, lighting = parts

            img = preprocess(
                Image.open(os.path.join(renders_dir, fname))
            ).unsqueeze(0).to(DEVICE)

            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
            embeddings.append(emb.cpu().numpy()[0])
            category_labels.append(category)
            lighting_labels.append(lighting)

    return (
        np.array(embeddings),
        np.array(category_labels),
        np.array(lighting_labels)
    )


def run_probe(embeddings, labels, probe_name):
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test) * 100

    print(f"{probe_name} accuracy: {acc:.1f}%")
    return acc


if __name__ == '__main__':
    embeddings, category_labels, lighting_labels = load_renders(RENDERS_DIR)
    print(f"Embeddings shape: {embeddings.shape}")
    print()

    cat_acc = run_probe(embeddings, category_labels, "Category (object)")
    lit_acc = run_probe(embeddings, lighting_labels, "Lighting (color)")

    print()
    print("=== Phase 2 Summary ===")
    print(f"Category accuracy:  {cat_acc:.1f}%")
    print(f"Lighting accuracy:  {lit_acc:.1f}%")
    print()
    if lit_acc > 50:
        print("Finding: CLIP embeddings encode lighting color — shortcut signal present")
    else:
        print("Finding: CLIP embeddings are robust to lighting color")