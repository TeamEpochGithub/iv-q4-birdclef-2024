"""Submit.py is the main script for running inference on the test set and creating a submission."""
import os
import warnings
from pathlib import Path

import hydra
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from src.config.submit_config import SubmitConfig
from src.setup.setup_data import setup_inference_data
from src.setup.setup_pipeline import setup_pipeline
from src.setup.setup_runtime_args import setup_pred_args
from src.utils.logger import logger
from src.utils.to_submission_format import to_submission_format

warnings.filterwarnings("ignore", category=UserWarning)

# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_submit", node=SubmitConfig)


@hydra.main(version_base=None, config_path="conf", config_name="submit")
# TODO(Epoch): Use SubmitConfig instead of DictConfig
def run_submit(cfg: DictConfig) -> None:
    """Run the main script for submitting the predictions."""
    print_section_separator("Q4 - BirdCLEF - Submit")

    # Set up logging
    import coloredlogs

    coloredlogs.install()

    # Get output directory
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Preload the pipeline
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, is_train=False)

    # Load the test data
    X = setup_inference_data(cfg.data_path)

    # Predict on the test data
    logger.info("Making predictions...")
    pred_args = setup_pred_args(pipeline=model_pipeline, output_dir=output_dir.as_posix(), data_dir=cfg.data_path, species_dir=cfg.species_path)
    predictions = model_pipeline.predict(X, **pred_args)

    # Make submission
    if predictions is not None:
        # Create a dataframe from the predictions

        submission = to_submission_format(predictions, cfg.data_path, cfg.species_path)

        # Print submission head
        logger.info(submission.head())

        # Save submissions to path (Might be different for other platforms than Kaggle)
        result_path = Path(cfg.result_path)
        os.makedirs(result_path, exist_ok=True)
        submission_path = result_path / "submission.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    else:
        raise ValueError("Predictions are None")


if __name__ == "__main__":
    run_submit()
