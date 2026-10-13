"""precomp.ml - learned springback prediction and surrogate-driven pre-compensation.

The target is the vertical deviation dz(x, y) = z_formed - z_commanded [m] of
the tool-side surface after springback, as a function of the COMMANDED surface,
the process and the material; the models are queried at compensated shapes
during displacement adjustment, so the data sets contain such shapes too.

Modules
-------
features     the versioned per-node feature schema (FEATURE_SCHEMA_VERSION,
             FeatureSpec, FeatureConfig, feature_maps)
dataset      Sample, the on-disk Dataset, feature tables, grouped splits
generate     design of experiments, commanded-shape variants, SparlabSimulator,
             ProxySimulator (analytic, NOT physics), data generation
models       GBMEnsemble, MLPEnsemble, ResidualModel, TransferModel
uncertainty  grouped split-conformal calibration
ood          the training envelope (Mahalanobis + kNN, point and part level)
surrogate    DeviationSurrogate (model + features + calibration + envelope),
             train_surrogate, transfer_surrogate
registry     save_model / load_model: bundles with manifests
evaluate     evaluation on held-out parts
compensate   surrogate displacement adjustment, surrogate_optimize

Every metric states its data source: "SparLab simulation", "proxy - not
physics" or "scan". torch is needed only for MLPEnsemble.
"""

from .compensate import (SurrogateCompensation, SurrogatePredictor, correction_basis,
                         surrogate_compensate, surrogate_optimize)
from .dataset import (SOURCE_LABELS, Dataset, Sample, Table, build_table, family_split,
                      grouped_split, source_label)
from .evaluate import EvaluationReport, evaluate_surrogate
from .features import (FEATURE_SCHEMA_VERSION, FeatureConfig, FeatureSpec, feature_maps,
                       global_features, region_labels)
from .generate import (DesignPoint, DesignSpace, PerturbationSpec, ProxyParams, ProxySimulator,
                       SparlabSimulator, design_points, generate, simulate_samples)
from .models import (FEAPrior, GBMEnsemble, MLPEnsemble, ResidualModel, TransferModel,
                     model_from_state)
from .ood import OODEnvelope
from .registry import load_model, read_manifest, save_model
from .surrogate import DeviationSurrogate, train_surrogate, transfer_surrogate
from .uncertainty import ConformalCalibrator

__all__ = [
    "FEATURE_SCHEMA_VERSION", "FeatureConfig", "FeatureSpec", "feature_maps",
    "global_features", "region_labels",
    "Sample", "Dataset", "Table", "build_table", "grouped_split",
    "family_split", "source_label", "SOURCE_LABELS",
    "DesignSpace", "DesignPoint", "design_points", "PerturbationSpec", "ProxyParams",
    "ProxySimulator", "SparlabSimulator", "generate", "simulate_samples",
    "GBMEnsemble", "MLPEnsemble", "ResidualModel", "TransferModel", "FEAPrior",
    "model_from_state",
    "ConformalCalibrator", "OODEnvelope",
    "DeviationSurrogate", "train_surrogate", "transfer_surrogate",
    "save_model", "load_model", "read_manifest",
    "EvaluationReport", "evaluate_surrogate",
    "SurrogatePredictor", "SurrogateCompensation", "surrogate_compensate",
    "surrogate_optimize", "correction_basis",
]
