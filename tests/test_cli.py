"""Tests for the package CLI."""

import json

from distributed_random_forest.cli import main


def test_quickstart_cli_json_output(capsys):
    """The CLI should emit a JSON report when requested."""
    exit_code = main(
        [
            '--samples', '200',
            '--features', '10',
            '--clients', '3',
            '--trees', '6',
            '--backend', 'sequential',
            '--json',
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert 'selected_strategy' in payload
