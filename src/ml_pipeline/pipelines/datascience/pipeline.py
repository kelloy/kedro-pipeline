"""
This is a boilerplate pipeline 'datascience'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_fine_tune, plot_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func= model_fine_tune,
            inputs = ["train_dataset","train_dataloader","test_dataset","test_dataloader",  "params:model_options"],
            outputs = ["best_state_dict","history"],
            name = "model_finetune_node"
        ),
        node(
            func = plot_metrics,
            inputs = ["history"],
            outputs = "plot",
            name = "visualize_metrics"
        )
    ])
