"""Testing NeMo pipeline where speech is synthesized from the text and then transcribed back to text."""

from logging import getLogger
from pathlib import Path

from deeppavlov import build_model
from deeppavlov.utils.pip_wrapper.pip_wrapper import install_from_config

log = getLogger(__name__)
tests_dir = Path(__file__).parent
speech_file_name = 'speech.wav'
speech_file_path = tests_dir / speech_file_name


def setup_module():
    if speech_file_path.exists():
        speech_file_path.unlink()


def teardown_module():
    setup_module()


class TestNemo:
    def setup(self):
        config_path = tests_dir / 'test_configs' / 'nemo' / 'tts2asr_test.json'
        install_from_config(config_path)
        self.nemo = build_model(config_path, download=True)

    def test_transcription(self):
        responses = []
        saved_file_path = None
        # Model is infered three times due to minor transcription errors
        for _ in range(3):
            transcription, saved_file_path = self.nemo(['string'], [speech_file_name])
            responses.extend(transcription)
        saved_file_path = Path(saved_file_path[0]).resolve()
        assert saved_file_path is not None, f'Saved file path is None. Responses are: {responses}'
        assert saved_file_path.exists(), f'The saved file does not exist. Responses are: {responses}'
        assert saved_file_path == speech_file_path.resolve(), f'save_file: {saved_file_path}, '\
                                                              f'speech_file: {speech_file_path}'
        assert 'string' in responses, f'There is no "string" in the responses. Responses are: {responses}'
