import pytest
from deeppavlov.core.commands.infer import end_repl_mode, interact_model, preparing_arguments


def test_end_repl_mode_decorator_keyboard_interrupt():
    """Check Ctrl-C."""

    def error_keyboard_interrupt():
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as ex:
        function = end_repl_mode(error_keyboard_interrupt)
        function()
    assert ex.value.code == 0


def test_end_repl_mode_decorator_eoferror():
    """Check Ctrl-D."""

    def error_eoferror():
        raise EOFError

    with pytest.raises(SystemExit) as ex:
        function = end_repl_mode(error_eoferror)
        function()
    assert ex.value.code == 0


def test_preparing_arguments(monkeypatch):
    """Check format arguments."""

    def input_data(data: str):
        return "data"

    monkeypatch.setattr("builtins.input", input_data)
    assert preparing_arguments([1, 2]) == [("data",), ("data",)]


def test_preparing_arguments_exit(monkeypatch):
    """Check exit by `q`"""

    def input_data(data: str):
        return "q"

    monkeypatch.setattr("builtins.input", input_data)
    with pytest.raises(SystemExit) as ex:
        preparing_arguments([1, 2])
    assert ex.value.code == 0
