"""Tests for main application entry point."""

from unittest.mock import patch

import pytest

from alpha_evolve_essay import main


def test_main_prints_welcome_message(capsys):
    """Test that main function prints the expected welcome message."""
    main()
    captured = capsys.readouterr()
    assert "AlphaEvolve Essay Writer - Coming Soon!" in captured.out


def test_main_runs_without_error():
    """Test that main function runs without raising exceptions."""
    try:
        main()
    except Exception as e:
        pytest.fail(f"main() raised an exception: {e}")


@patch("builtins.print")
def test_main_calls_print(mock_print):
    """Test that main function calls print with correct message."""
    main()
    mock_print.assert_called_once_with("AlphaEvolve Essay Writer - Coming Soon!")
