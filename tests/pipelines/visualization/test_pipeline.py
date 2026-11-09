import pytest
from matplotlib.figure import Figure

from netflix.pipelines.data_processing.nodes import clean_data
from netflix.pipelines.visualization.nodes import plot_eda

PARAMS = {"id_column": "customer_id", "target_column": "churned", "numeric_bins": 4}


@pytest.fixture
def clean_customers(raw_customers, dp_params):
    return clean_data(raw_customers, dp_params)


def test_plot_eda_returns_all_named_figures(clean_customers):
    figures = plot_eda(clean_customers, PARAMS)

    assert set(figures) == {
        "target_balance.png",
        "numeric_distributions.png",
        "churn_rate_numeric.png",
        "churn_rate_categorical.png",
        "correlation_heatmap.png",
    }
    assert all(isinstance(fig, Figure) for fig in figures.values())


def test_figures_can_be_rendered_to_png(clean_customers, tmp_path):
    figures = plot_eda(clean_customers, PARAMS)

    for name, fig in figures.items():
        fig.savefig(tmp_path / name)

    assert all((tmp_path / name).stat().st_size > 0 for name in figures)
