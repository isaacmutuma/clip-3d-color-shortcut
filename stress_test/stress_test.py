import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import clip
import numpy as np
from PIL import Image

RENDERS_DIR = 'renders'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'car', 'chair', 'cone', 'bowl'
]

LIGHTING_CONDITIONS = ['neutral', 'warm', 'cool', 'green']


def zero_shot_accuracy(model, preprocess, renders_dir, lighting):
    # encode text labels
    text_tokens = clip.tokenize(
        [f"a 3D rendering of a {c}" for c in CATEGORIES]
    ).to(DEVICE)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct = 0
    total = 0

    files = [
        f for f in sorted(os.listdir(renders_dir))
        if f.endswith(f'__{lighting}.png')
    ]

    for fname in files:
        parts = fname.replace('.png', '').split('__')
        if len(parts) != 3:
            continue
        true_category = parts[0]
        if true_category not in CATEGORIES:
            continue

        img = preprocess(
            Image.open(os.path.join(renders_dir, fname))
        ).unsqueeze(0).to(DEVICE)

        img_features = model.encode_image(img)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        similarities = (img_features @ text_features.T).squeeze(0)
        pred_idx = similarities.argmax().item()
        pred_category = CATEGORIES[pred_idx]

        if pred_category == true_category:
            correct += 1
        total += 1

    acc = correct / total * 100 if total > 0 else 0
    return acc


if __name__ == '__main__':
    model, preprocess = clip.load('ViT-B/32', device=DEVICE)
    model.eval()

    print("=== Phase 3 — Zero-Shot Stress Test ===")
    print(f"{'Lighting':<12} {'Accuracy':>10}")
    print("-" * 24)

    results = {}
    for lighting in LIGHTING_CONDITIONS:
        acc = zero_shot_accuracy(model, preprocess, RENDERS_DIR, lighting)
        results[lighting] = acc
        print(f"{lighting:<12} {acc:>9.1f}%")

    print()
    baseline = results['neutral']
    print(f"Baseline (neutral): {baseline:.1f}%")
    for lighting in ['warm', 'cool', 'green']:
        drop = results[lighting] - baseline
        print(f"Drop under {lighting}: {drop:+.1f}%")