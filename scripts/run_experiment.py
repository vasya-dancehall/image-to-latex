from argparse import Namespace
from typing import List, Optional

import wandb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from image_to_latex.data import Im2Latex
from image_to_latex.lit_models import LitResNetTransformer


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    wandb.login(key="138ca1cc6b82a08a807dd42fa96a2514d1fc16a5")
    datamodule = Im2Latex(**cfg.data)
    datamodule.setup()

    lit_model = LitResNetTransformer(**cfg.lit_model)
    lit_model = lit_model.load_from_checkpoint("/image2latex/artifacts/model.pt")

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(lit_model, datamodule=datamodule)
    trainer.fit(lit_model, datamodule=datamodule)
    trainer.test(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
