"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import label_mapping, create_dataset, create_data_loader,train_test_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= label_mapping,
                inputs = "data",
                outputs = "preprocessed_data",
                name = "preprocessed_data_node"
            ),

            node(
                func = create_dataset,
                inputs = ["train_data","test_data"],
                outputs = ["train_dataset","test_dataset"],
                name = "dataset_node",
            ),

            node(
                func = create_data_loader,
                inputs = ["train_dataset","test_dataset"],
                outputs= ["train_dataloader","test_dataloader"],
                name = "dataloader_node"
            ),

            node(
                func = train_test_split,
                inputs = ["preprocessed_data"],
                outputs = ["train_data","test_data"],
                name = "train_test_split_node"
            )
    ])
