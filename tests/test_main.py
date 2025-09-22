import pytest
from trodestrack import main


def test_main_smoke():
    """Smoke test for main() function - should run without errors."""
    main()


def test_main_function_exists():
    """Test that main function is callable."""
    assert callable(main)