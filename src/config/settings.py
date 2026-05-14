from __future__ import annotations

import os
from dataclasses import dataclass, field


def _project_root() -> str:
    # src/config/settings.py -> ../../
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


@dataclass(frozen=True)
class PathSettings:
    root: str = field(default_factory=_project_root)
    data: str = field(init=False)
    models: str = field(init=False)
    plots: str = field(init=False)
    reports: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data", os.path.join(self.root, "data"))
        object.__setattr__(self, "models", os.path.join(self.root, "saved_models"))
        object.__setattr__(self, "plots", os.path.join(self.root, "plots"))
        object.__setattr__(self, "reports", os.path.join(self.root, "reports"))

        for path in [self.data, self.models, self.plots, self.reports]:
            os.makedirs(path, exist_ok=True)

    @property
    def raw_data(self) -> str:
        return os.path.join(self.data, "labeled_data.csv")

    @property
    def model_checkpoint(self) -> str:
        return os.path.join(self.models, "sbert_transformer_clf.pt")

    @property
    def tokenizer_dir(self) -> str:
        return os.path.join(self.models, "tokenizer")


@dataclass(frozen=True)
class LabelSettings:
    class_names: dict = field(default_factory=lambda: {
        0: "hate_speech", 1: "offensive_language", 2: "neither"
    })
    binary_map: dict = field(default_factory=lambda: {0: 1, 1: 1, 2: 0})
    binary_labels: dict = field(default_factory=lambda: {0: "clean", 1: "offensive"})


@dataclass(frozen=True)
class SplitSettings:
    random_seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15


@dataclass(frozen=True)
class ModelSettings:
    sbert_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    sbert_max_seq_len: int = 128
    sbert_hidden_dim: int = 768
    sbert_batch_size: int = 32
    # TransformerEncoder
    tc_num_heads: int = 8
    tc_num_layers: int = 3
    tc_ffn_dim: int = 2048
    tc_dropout: float = 0.2
    tc_attn_dropout: float = 0.1
    tc_head_hidden_dims: tuple = (512, 256)
    tc_head_dropout: float = 0.3


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    early_stop_patience: int = 4
    # Focal Loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass(frozen=True)
class MLflowSettings:
    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "offensive-text-detection")
    )
    run_name: str = ""
    log_model: bool = True
    log_plots: bool = True
    tags: dict = field(default_factory=lambda: {
        "model": "MPNet + TransformerEncoder",
        "framework": "PyTorch",
        "task": "offensive-text-detection",
        "dataset": "Hate Speech and Offensive Language Dataset"
    })


@dataclass(frozen=True)
class InferenceSettings:
    classification_threshold: float = 0.5


@dataclass(frozen=True)
class Settings:
    paths: PathSettings = field(default_factory=PathSettings)
    labels: LabelSettings = field(default_factory=LabelSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    mlflow: MLflowSettings = field(default_factory=MLflowSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)