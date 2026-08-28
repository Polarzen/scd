from src.cli import main


def test_no_args_only_prints_help(capsys):
    assert main([]) == 0
    captured = capsys.readouterr()
    assert "validate" in captured.out
    assert "repeated-cv" in captured.out
