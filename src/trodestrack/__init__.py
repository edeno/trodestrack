def main() -> None:
    """Entry point for trodestrack CLI."""
    import sys
    from .cli.main import main as cli_main
    sys.exit(cli_main())