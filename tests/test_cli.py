"""
Smoke tests for CLI.
"""
import subprocess
import sys


def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        [sys.executable, "-m", "src.spark_trainer.cli", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "SparkTrainer" in result.stdout


def test_cli_validate():
    """Test CLI validate command."""
    result = subprocess.run(
        [sys.executable, "-m", "src.spark_trainer.cli", "validate"],
        capture_output=True,
        text=True,
    )

    # May fail if CUDA not available, but should not crash
    assert result.returncode in [0, 1]


def test_cli_train_missing_args():
    """Test CLI train command with missing arguments."""
    result = subprocess.run(
        [sys.executable, "-m", "src.spark_trainer.cli", "train"],
        capture_output=True,
        text=True,
    )

    # Should fail due to missing required arguments
    assert result.returncode == 1
