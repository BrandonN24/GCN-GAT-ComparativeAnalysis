import sys
from pathlib import Path

# Ensure repository root is on sys.path so `training` package imports work during tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide lightweight fake modules for heavy dependencies so importing the module
# doesn't require torch/torch_geometric to be installed during tests.
import types

fake_torch = types.ModuleType("torch")
sys.modules["torch"] = fake_torch

tg = types.ModuleType("torch_geometric")
tg_datasets = types.ModuleType("torch_geometric.datasets")
tg_transforms = types.ModuleType("torch_geometric.transforms")

def _fake_Planetoid(root, name, transform=None):
    # This will be overridden in the actual test via monkeypatch, but provide a
    # default placeholder to satisfy imports.
    return None

tg_datasets.Planetoid = _fake_Planetoid
tg_transforms.NormalizeFeatures = lambda: None

sys.modules["torch_geometric"] = tg
sys.modules["torch_geometric.datasets"] = tg_datasets
sys.modules["torch_geometric.transforms"] = tg_transforms

import pytest

from training import citeseer


class DummyDataset:
    def __init__(self, root, name, transform):
        self.root = root
        self.name = name
        self.transform = transform
        self.num_classes = 3

    def __getitem__(self, idx):
        return {"x": "node_features", "edge_index": "edges", "y": 0}

    def __len__(self):
        return 1


def test_load_citeseer_monkeypatched(monkeypatch):
    """Ensure load_citeseer returns dataset[0] and num_classes when Planetoid is patched."""

    def fake_Planetoid(root, name, transform):
        return DummyDataset(root, name, transform)

    monkeypatch.setattr(citeseer, "Planetoid", fake_Planetoid)

    data, num_classes = citeseer.load_citeseer()

    assert isinstance(data, dict)
    assert data["x"] == "node_features"
    assert data["edge_index"] == "edges"
    assert data["y"] == 0
    assert num_classes == 3
