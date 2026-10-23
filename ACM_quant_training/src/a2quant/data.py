"""Frozen SIRST splits; ACM reference augmentation with explicit binary masks."""

from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset


class SirstDataset(Dataset):
    def __init__(self, root, ids, training=False, size=512):
        self.root, self.ids, self.training, self.size = Path(root), list(ids), training, size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        with Image.open(self.root / "images" / (name + ".png")) as file:
            image = file.convert("RGB")
        with Image.open(self.root / "masks" / (name + ".png")) as file:
            mask = file.convert("L")
        if self.training:
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            long_size = random.randint(int(self.size * .5), int(self.size * 2.))
            width, height = image.size
            if height > width:
                height, width = long_size, int(width * long_size / height + .5)
            else:
                width, height = long_size, int(height * long_size / width + .5)
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            padding = (0, 0, max(0, self.size - width), max(0, self.size - height))
            image, mask = ImageOps.expand(image, border=padding), ImageOps.expand(mask, border=padding)
            x, y = random.randint(0, image.width - self.size), random.randint(0, image.height - self.size)
            box = (x, y, x + self.size, y + self.size)
            image, mask = image.crop(box), mask.crop(box)
            if random.random() < .5:
                image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))
        else:
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255.
        array = (array - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]) / np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
        target = (np.asarray(mask) > 0).astype(np.float32)[None].copy()
        return torch.from_numpy(array), torch.from_numpy(target), name


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
