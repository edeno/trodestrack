import pytest
from trodestrack import main


def test_main_smoke():
    """Smoke test for main() function - should handle missing subcommand gracefully."""
    # The CLI now requires a subcommand, so calling main() without args should exit
    with pytest.raises(SystemExit):
        main()


def test_main_function_exists():
    """Test that main function is callable."""
    assert callable(main)