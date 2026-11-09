"""Pipeline 'visualization': exploratory charts saved under data/08_reporting."""

from kedro.pipeline import Node, Pipeline

from .nodes import plot_eda


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                plot_eda,
                inputs=["netflix_clean", "params:visualization"],
                outputs="eda_figures",
                name="plot_eda",
            ),
        ]
    )
