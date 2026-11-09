"""Pipeline 'data_processing': netflix_raw -> netflix_clean."""

from kedro.pipeline import Node, Pipeline

from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                clean_data,
                inputs=["netflix_raw", "params:data_processing"],
                outputs="netflix_clean",
                name="clean_data",
            ),
        ]
    )
