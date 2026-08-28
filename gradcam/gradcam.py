import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

RENDERS_DIR = 'renders'
OUTPUT_DIR  = 'attention_outputs'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'car', 'chair', 'cone', 'bowl'
]
LIGHTING_CONDITIONS = ['neutral', 'warm', 'cool', 'green']

model, preprocess = clip.load('ViT-B/32', device=DEVICE)
model = model.float()
model.eval()

def get_attention_map(model, img_tensor):
    captured = []

    def hook_fn(module, input, output):
        captured.append(output.detach())

    handle = model.visual.transformer.resblocks[-1].ln_2.register_forward_hook(hook_fn)

    with torch.no_grad():
        model.encode_image(img_tensor)

    handle.remove()

    if not captured:
        return None

    tokens = captured[0].squeeze(1)
    if tokens.dim() == 3:
        tokens = tokens[:, 0, :]

    cls     = tokens[0]
    patches = tokens[1:]

    cls_norm   = cls / cls.norm()
    patch_norm = patches / patches.norm(dim=-1, keepdim=True)
    similarity = (patch_norm @ cls_norm).cpu().numpy()

    attn_map = similarity.reshape(7, 7)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


if __name__ == '__main__':
    for category in CATEGORIES:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'CLIP Attention — {category}', fontsize=14)

        for idx, lighting in enumerate(LIGHTING_CONDITIONS):
            matches = [
                f for f in os.listdir(RENDERS_DIR)
                if f.startswith(f'{category}__') and f.endswith(f'__{lighting}.png')
            ]
            if not matches:
                axes[idx].axis('off')
                continue

            img_path = os.path.join(RENDERS_DIR, matches[0])
            raw_img  = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(raw_img).unsqueeze(0).to(DEVICE).float()

            try:
                attn_map = get_attention_map(model, img_tensor)
                if attn_map is None:
                    raise ValueError("Empty")

                rgb = np.array(raw_img.resize((224, 224))) / 255.0
                attn_up = np.array(
                    Image.fromarray((attn_map * 255).astype(np.uint8))
                    .resize((224, 224), Image.BILINEAR)
                ) / 255.0
                heatmap = plt.cm.jet(attn_up)[:, :, :3]
                overlay = np.clip(0.5 * rgb + 0.5 * heatmap, 0, 1)

                axes[idx].imshow(overlay)
                axes[idx].set_title(lighting)
                axes[idx].axis('off')
            except Exception as e:
                print(f"Failed {category}/{lighting}: {e}")
                axes[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'{category}_attention.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print("Done")