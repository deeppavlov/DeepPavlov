import pytest
from deeppavlov.core.commands.infer import end_repl_mode


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
