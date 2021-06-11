import logging
import os
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)


def get_checkpoint_callback(output_dir, metric):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "sari_avgaddscore":
        exp = "{val_avg_sari_avgaddscore:.4f}-{step_count}"
    elif metric == "accuracy":
        exp = "{val_avg_accuracy:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2 and bleu, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="max",
        save_top_k=1,
        period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback