"""
End-to-end test: the default pipeline runs on the real raw data and writes metrics.
Unit tests of individual nodes live in ``tests/pipelines``.
"""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    def test_kedro_run_default_pipeline(self):
        project_path = Path(__file__).resolve().parents[1]
        bootstrap_project(project_path)

        with KedroSession.create(project_path=project_path) as session:
            session.run()

        assert (project_path / "data/08_reporting/metrics.json").exists()
