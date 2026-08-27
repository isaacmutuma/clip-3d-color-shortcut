import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import trimesh
import pyrender
import numpy as np
from PIL import Image

# --- config ---
MODELNET_DIR = 'ModelNet40'
OUTPUT_DIR   = 'renders'
IMG_SIZE     = 224
N_PER_CLASS  = 10

CATEGORIES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'car', 'chair', 'cone', 'bowl'
]

LIGHTING = {
    'neutral': (1.0, 1.0, 1.0),
    'warm':    (1.0, 0.6, 0.3),
    'cool':    (0.3, 0.6, 1.0),
    'green':   (0.3, 1.0, 0.3),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def render_mesh(mesh_path, light_color):
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)

    scene = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    scene.add(camera, pose=cam_pose)

    r, g, b = light_color
    light = pyrender.DirectionalLight(
        color=np.array([r, g, b]), intensity=3.0
    )
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(IMG_SIZE, IMG_SIZE)
    color, _ = renderer.render(scene)
    renderer.delete()

    return Image.fromarray(color)


if __name__ == '__main__':
    total = 0
    for category in CATEGORIES:
        train_dir = os.path.join(MODELNET_DIR, category, 'train')
        mesh_files = sorted([
            f for f in os.listdir(train_dir)
            if f.endswith('.off')
        ])[:N_PER_CLASS]

        for mesh_file in mesh_files:
            mesh_path = os.path.join(train_dir, mesh_file)
            stem = mesh_file.replace('.off', '')

            for light_name, light_color in LIGHTING.items():
                try:
                    img = render_mesh(mesh_path, light_color)
                    fname = f"{category}__{stem}__{light_name}.png"
                    img.save(os.path.join(OUTPUT_DIR, fname))
                    total += 1
                except Exception as e:
                    print(f"Failed {mesh_file} / {light_name}: {e}")

        print(f"Done: {category}")

    print(f"\nTotal renders saved: {total}")