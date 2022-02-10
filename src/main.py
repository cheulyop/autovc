from os import path
from typing import List, Optional, Type

import hydra
import pytorch_lightning as pl
import transformers
from omegaconf import DictConfig
from pytorch_lightning.loggers import LightningLoggerBase
from transformers import TrainingArguments

from src.base import Pipeline
from src.utils import utils

LOG = utils.get_logger(__name__)


def body(config: DictConfig) -> Optional[float]:
    """Contains training/preprocessing pipeline.
    Instantiates all required objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        pl.seed_everything(config.seed, workers=True)

    # If pipeline was given in configuration, run it
    if config.get("pipeline") is not None:
        LOG.info(f"Instantiating pipeline <{config.pipeline._target_}>")
        pipeline: Type[Pipeline] = hydra.utils.instantiate(config.pipeline)
        LOG.info("Running pipeline!")
        pipeline.run()

    else:
        # Init lightning datamodule
        LOG.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

        # Init generic model
        LOG.info(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        # Load pretrained model from checkpoint if available
        if (
            config.get("from_pretrained") is not None
        ):  # NOTE: get defaults to None when key is not in DictConfig
            print(config.get("from_pretrained"))
            model = model.load_from_checkpoint(**config.get("from_pretrained"))

        # Prepare and setup datamodule prior to training
        if config.trainer._target_ == "pytorch_lightning.Trainer":
            datamodule.setup()

            fold_idx = config.fold_idx
            datamodule.set_folds(fold_idx)
            logger: List[LightningLoggerBase] = []
            if "logger" in config:
                for _, lg_conf in config.logger.items():
                    if "_target_" in lg_conf:
                        LOG.info(f"Instantiating logger <{lg_conf._target_}>")
                        lg_conf.name = "-".join(lg_conf.name.split("-")[:-1]) + f"-fold{fold_idx}"
                        logger.append(hydra.utils.instantiate(lg_conf))

            # Init lightning callbacks
            callbacks: List[pl.Callback] = []
            if "callbacks" in config:
                for _, cb_conf in config.callbacks.items():
                    if "_target_" in cb_conf:
                        LOG.info(f"Instantiating callback <{cb_conf._target_}>")

                        if "filename" in cb_conf:
                            cb_conf.filename = (
                                "-".join(cb_conf.filename.split("-")[:-1]) + f"_fold{fold_idx}"
                            )
                        callbacks.append(hydra.utils.instantiate(cb_conf))

            trainer: pl.Trainer = hydra.utils.instantiate(
                config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
            )

            if not config.get("skip_training"):
                LOG.info("Starting training!")
                trainer.fit(model=model, datamodule=datamodule)
            else:
                LOG.info("Skipping training")
            # Evaluate model on test set, using the best model achieved during training
            if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
                LOG.info("Starting testing!")
                trainer.test(
                    model,
                    test_dataloaders=datamodule.test_dataloader(),
                    ckpt_path="best",
                )

            if (
                config.get("cross_validation")
                and config.cross_validation.save_dir is not None
                and not config.get("skip_training")
            ):
                LOG.info(f"Fold {fold_idx}: Saving predictions and true labels")
                utils.save_preds_and_labels(
                    save_dir=config.cross_validation.save_dir,
                    fname=f"{config.cross_validation.savefile}_{fold_idx}.pkl",
                    getter_fn=model.get_preds_and_labels,
                    val_dataloader=datamodule.val_dataloader(),
                    test_dataloader=datamodule.test_dataloader(),
                )

                # when get predictions seperately
                # LOG.info("get predictions for CV")
                # model.get_predicts(val_dl=datamodule.val_dataloader(), test_dl=datamodule.test_dataloader(), fold_index=4
                #                   ,cv_result_dir=config.model.cv_result_dir)

        # Init huggingface trainer
        elif config.trainer._target_ == "transformers.Trainer":
            # Setup datamodule
            if config.datamodule.get("run_prepare_data") is True:
                datamodule.prepare_data()
            datamodule.setup()

            LOG.info(f"Instantiating training_args <{config.trainer.args._target_}>")
            training_args: TrainingArguments = hydra.utils.instantiate(
                config.trainer.args,
                output_dir=path.join(datamodule.cache_dir, "models"),
            )
            LOG.info(f"Instantiating metrics <{config.trainer.metrics._target_}>")
            metrics: HFMetric = hydra.utils.instantiate(
                config.trainer.metrics,
            )
            LOG.info(f"Instantiating trainer <{config.trainer._target_}>")
            trainer: transformers.Trainer = transformers.Trainer(  # FIXME: non optional field (eval_steps) cannot be assigned None when instantiating with hydra.utils.instantiate? why?
                model=model,
                args=training_args,
                train_dataset=datamodule.train_set,
                eval_dataset=datamodule.val_set,
                compute_metrics=metrics.compute,
            )

            # Train the model
            LOG.info("Starting training!")
            trainer.train()

            # Evaluate model on test set, using the best model achieved during training
            if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
                LOG.info("Starting testing!")
                trainer.evaluate(eval_dataset=datamodule.test_set)

        # Send some parameters from config to all lightning loggers
        LOG.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Make sure everything closed properly
        LOG.info("Finalizing!")
        utils.finish(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

        # Print path to best checkpoint
        LOG.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        optimized_metric = config.get("optimized_metric")
        if optimized_metric:
            return trainer.callback_metrics[optimized_metric]
