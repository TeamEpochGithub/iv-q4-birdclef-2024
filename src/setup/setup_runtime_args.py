"""File containing functions related to setting up runtime arguments for pipelines."""

from typing import Any

from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.model import ModelPipeline


def setup_train_args(
    pipeline: ModelPipeline | EnsemblePipeline,
    cache_args: dict[str, Any],
    train_indices: list[int] | dict[str, list[int]],
    test_indices: list[int] | dict[str, list[int]],
    fold: int = -1,
    *,
    save_model: bool = False,
    save_model_preds: bool = False,
) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :param cache_args: Caching arguments
    :param train_indices: Train indices
    :param test_indices: Test indices
    :param fold: Fold number if it exists
    :param save_model: Whether to save the model to File
    :param save_model_preds: Whether to save the model predictions
    :return: Dictionary containing arguments
    """
    # Main trainer arguments
    main_trainer = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "save_model": save_model,
    }

    if fold > -1:
        main_trainer["fold"] = fold  # type: ignore[assignment]

    # Train system arguments
    train_sys = {
        "MainTrainer": main_trainer,
    }

    if save_model_preds:
        train_sys["cache_args"] = cache_args

    train_args: dict[str, dict[str, Any]] = {
        "x_sys": {},
        "y_sys": {},
        "train_sys": train_sys,
    }

    if isinstance(pipeline, EnsemblePipeline):
        train_args = {
            "ModelPipeline": train_args,
        }

    return train_args


def setup_pred_args(pipeline: ModelPipeline | EnsemblePipeline, output_dir: str, data_dir: str, species_dir: str) -> dict[str, Any]:
    """Set train arguments for pipeline.

    :param pipeline: Pipeline to receive arguments
    :param output_dir: Output directory
    :param data_dir: Data directory
    :param species_dir: Species directory
    :return: Dictionary containing arguments
    """
    pred_args: dict[str, Any] = {
        "train_sys": {
            "MainTrainer": {
                "batch_size": 1,
            },
            "VisualizePreds": {
                "output_dir": output_dir,
                "data_dir": data_dir,
                "species_dir": species_dir,
            },
        },
    }
    # pred_args: dict[str, Any] = {}

    if isinstance(pipeline, EnsemblePipeline):
        pred_args = {
            "ModelPipeline": pred_args,
        }

    return pred_args
