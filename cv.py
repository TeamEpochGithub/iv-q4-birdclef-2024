"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""
import copy
import gc
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import hydra
import randomname
import wandb
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy.typing as npt

from src.config.cv_config import CVConfig
from src.setup.setup_data import setup_inference_data, setup_train_x_data, setup_train_y_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_pred_args, setup_train_args
from src.setup.setup_wandb import setup_wandb
from src.typing.typing import YData
from src.utils.lock import Lock
from src.utils.logger import logger
from src.utils.set_torch_seed import set_torch_seed

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Run the cv config with an optional lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_cv_cfg(cfg)

def collate_fn(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch
    return X, y

def custom_predict(self, x: Any, **pred_args: Any) -> npt.NDArray[np.float32]:  # noqa: ANN401
        """Predict on the test data.

        :param x: The input to the system.
        :return: The output of the system.
        """
        print_section_separator(f"Predicting model: {self.model.__class__.__name__}")
        self.log_to_debug(f"Predicting model: {self.model.__class__.__name__}")

        # Parse pred_args
        curr_batch_size = pred_args.get("batch_size", self.batch_size)

        # Create dataset
        pred_dataset = self.create_prediction_dataset(x)
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=curr_batch_size,
            shuffle=False,
            collate_fn=(collate_fn if hasattr(pred_dataset, "__getitems__") else None),  # type: ignore[arg-type],
            # num_workers=16,
            # prefetch_factor=1
        )

        # Predict with a single model
        if self.n_folds < 1 or pred_args.get("use_single_model", False):
            self._load_model()
            return self.predict_on_loader(pred_dataloader)

        predictions = []
        # Predict with multiple models
        for i in range(int(self.n_folds)):
            self._fold = i  # set the fold, which updates the hash
            # Try to load the next fold if it exists
            try:
                self._load_model()
            except FileNotFoundError as e:
                if i == 0:
                    raise FileNotFoundError(f"First model of {self.n_folds} folds not found...") from e
                self.log_to_warning(f"Model for fold {self._fold} not found, skipping the rest of the folds...")
            self.log_to_terminal(f"Predicting with model fold {i + 1}/{self.n_folds}")
            predictions.append(self.predict_on_loader(pred_dataloader))

        # Average the predictions using numpy
        test_predictions = np.array(predictions)
        return test_predictions


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q4 - BirdCLEF - CV")

    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Set up Weights & Biases group name
    if cfg.wandb.enabled:
        wandb_group_name = randomname.get_name()
        setup_wandb(cfg, "cv", output_dir, name=wandb_group_name, group=wandb_group_name)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg)

    # Cache arguments for x_sys
    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    # Read the data if required and split in X, y
    x_cache_exists = model_pipeline.get_x_cache_exists(cache_args)
    # y_cache_exists = model_pipeline.get_y_cache_exists(cache_args)

    X = None
    if not x_cache_exists:
        X = setup_train_x_data(cfg.raw_path, cfg.years)

    y = setup_train_y_data(cfg.raw_path, cfg.years)

    # Instantiate scorer
    scorer = instantiate(cfg.scorer)
    scores: list[dict[str, float]] = []

    # Split indices into train and test
    # splitter_data = setup_splitter_data()
    logger.info("Using splitter to split data into train and test sets.")

    if not isinstance(y, YData):
        raise TypeError("Y Should be YData")
    
    # Will keep the preds of each fold fro unlabelled soundscapes
    fold_preds = []
    for fold_no, (train_indices, test_indices) in enumerate(instantiate(cfg.splitter).split(y)):
        copy_x = copy.deepcopy(X)

        score, predictions = run_fold(fold_no, X, y, train_indices, test_indices, cfg, scorer, output_dir, cache_args)
        scores.append(score)

        X = copy_x
        if score < 0.9:
            break

    # Set up the inference pipeline
    logger.info('Setting up the inference pipeline for unlabeled soundscapes')
    model_pipeline = setup_pipeline(cfg, is_train=False)
    model_pipeline.train_sys.steps[0].custom_predict = custom_predict.__get__(model_pipeline.train_sys.steps[0])
    data_path = "data/raw/2024/unlabeled_soundscapes/"
    X_unlabeled = setup_inference_data(data_path)
    X_unlabeled['bird_2024'] = X_unlabeled['bird_2024'][:1000]
    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    pred_args = setup_pred_args(pipeline=model_pipeline, output_dir=output_dir.as_posix(), data_dir=data_path, species_dir="data/raw/2024/train_audio/")
    
    logger.info('Making predictions on unlabeled soundscapes')
    predictions = model_pipeline.predict(X_unlabeled, **pred_args)
    fold_preds = predictions
    print(fold_preds.shape)

    # Find the pairs
    pairs = []
    for i in range(fold_no+1):
        pairs.extend(tuple(zip([i]*len(list(range(i+1, fold_no+1))),list(range(i+1, fold_no+1)))))

    corrs = {}
    # Find the pairwise correlation between all the arrays
    pairs.append((0,0))
    # make directory for output plots
    os.makedirs(output_dir/Path('fold_correlations'))
    for pair in pairs:
        corr = np.corrcoef(x=fold_preds[pair[0]], y=fold_preds[pair[1]],rowvar=False)
        corrs[pair] = corr[:182, 182:364]
        plt.figure(figsize=(12,12))
        sns.heatmap(corrs[pair])
        diag_corr = sum([corrs[pair][i,i] for i in range(corrs[pair].shape[0])]) / 182
        corrs[pair] = diag_corr
        plt.title(f'pair {pair} diag_corr:{diag_corr}')
        plt.savefig(output_dir/Path('fold_correlations')/Path(f'{pair}.png'))

    mean_corr = 0
    for pair in corrs:
        mean_corr += corrs[pair]
    mean_corr /= len(corrs)

    avg_score: dict[str, float]
    avg_score = {}
    for score in scores:
        for year in score:  # type: ignore[union-attr]
            if avg_score.get(year) is not None:
                avg_score[year] += score[year] / len(scores)
            else:
                avg_score[year] = score[year] / len(scores)

    print_section_separator("CV - Results")
    logger.info(f"Avg Score: {avg_score}")
    [wandb.log({f"Avg Score_{year}": avg_score[year]}) for year in avg_score] if isinstance(avg_score, dict) else wandb.log({"Avg Score": avg_score})
    wandb.log({"Score": avg_score["2024"]}) if isinstance(avg_score, dict) and "2024" in avg_score else None
    wandb.log({"Fold correlations": str(corrs)})
    wandb.log({"Mean corr": mean_corr})

    # sweep score is score 2024 + weight* mean_corr
    sweep_score = avg_score["2024"] + cfg.corr_weight * mean_corr
    wandb.log({"Sweepscore": sweep_score})

    logger.info("Finishing wandb run")
    wandb.finish()


def run_fold(
    fold_no: int,
    X: Any,  # noqa: ANN401
    y: Any,  # noqa: ANN401
    train_indices: list[int],
    test_indices: list[int],
    cfg: DictConfig,
    scorer: Any,  # noqa: ANN401
    output_dir: Path,
    cache_args: dict[str, Any],
) -> tuple[dict[str, float], Any]:
    """Run a single fold of the cross validation.

    :param i: The fold number.
    :param X: The input data.
    :param y: The labels.
    :param train_indices: The indices of the training data.
    :param test_indices: The indices of the test data.
    :param cfg: The config file.
    :param scorer: The scorer to use.
    :param output_dir: The output directory for the prediction plots.
    :param processed_y: The processed labels.
    :return: The score of the fold and the predictions.
    """
    # Print section separator
    print_section_separator(f"CV - Fold {fold_no}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg)

    train_args = setup_train_args(
        pipeline=model_pipeline,
        cache_args=cache_args,
        train_indices=train_indices,
        test_indices=test_indices,
        fold=fold_no,
        save_model=cfg.save_folds,
        save_model_preds=False,
    )

    predictions, _ = model_pipeline.train(X, y, **train_args)

    gc.collect()

    score = scorer(y_true=y, y_pred=predictions, test_indices=test_indices, years=cfg.years, output_dir=output_dir)
    logger.info(f"Score, fold {fold_no}: {score}")

    fold_dir = output_dir / str(fold_no)  # Files specific to a run can be saved here
    logger.debug(f"Output Directory: {fold_dir}")

    if wandb.run:
        [wandb.log({f"Score_{year}_{fold_no}": score[year]}) for year in score] if isinstance(score, dict) else wandb.log({f"Score_{fold_no}": score})
    return score, predictions


if __name__ == "__main__":
    run_cv()

