# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger

import librosa
import numpy as np
from nemo_tts import WaveGlowInferNM

log = getLogger(__name__)


class BaseVocoder:
    def __call__(self, tensor):
        raise NotImplementedError

    def get_audio(self, input_tensor, mel_len):
        raise NotImplementedError


class WaveGlow(BaseVocoder):
    """Wraps WaveGlowInferNM module."""
    def __init__(self, *,
                 denoiser_strength: float = 0.0,
                 n_window_stride: int = 160,
                 **kwargs) -> None:
        """Builds WaveGlowInferNM object.

        Args:
            denoiser_strength: Denoiser strength for waveglow.
            n_window_stride: Stride of window for fft in samples used in model training.
            kwargs: Named arguments for WaveGlowInferNM constructor.

        """
        self.waveglow = WaveGlowInferNM(**kwargs)
        self.denoiser_strength = denoiser_strength
        self.n_window_stride = n_window_stride

    def __call__(self, mel_postnet):
        return self.waveglow(mel_spectrogram=mel_postnet)

    def __str__(self):
        return str(self.waveglow)

    def restore_from(self, path):
        self.waveglow.restore_from(path)
        if self.denoiser_strength > 0:
            log.info('Setup denoiser for WaveGlow')
            self.waveglow.setup_denoiser()

    def get_audio(self, audio_tensor, mel_len):
        audios = []
        for i, batch in enumerate(audio_tensor):
            audio = batch.cpu().numpy()
            for j, sample in enumerate(audio):
                sample_len = mel_len[i][j] * self.n_window_stride
                sample = sample[:sample_len]
                if self.denoiser_strength > 0:
                    sample, _ = self.waveglow.denoise(sample, strength=self.denoiser_strength)
                audios.append(sample)
        return audios


class GriffinLim(BaseVocoder):
    def __init__(self, *,
                 sample_rate: float = 16000.0,
                 n_fft: int = 1024,
                 mag_scale: float = 2048.0,
                 power: float = 1.2,
                 n_iters: int = 50,
                 **kwargs):
        self.mag_scale = mag_scale
        self.power = power
        self.n_iters = n_iters
        self.n_fft = n_fft
        self.filterbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            **kwargs
        )

    def __call__(self, mel_postnet):
        return mel_postnet

    def get_audio(self, mel_spec, mel_len):
        audios = []
        for i, batch in enumerate(mel_spec):
            log_mel = batch.cpu().numpy().transpose(0, 2, 1)
            mel = np.exp(log_mel)
            magnitudes = np.dot(mel, self.filterbank) * self.mag_scale
            for j, sample in enumerate(magnitudes):
                sample = sample[:mel_len[i][j], :]
                audio = self.griffin_lim(sample.T ** self.power)
                audios.append(audio)
        return audios

    def griffin_lim(self, magnitudes):
        """
        Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
        """
        phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
        if not np.isfinite(signal).all():
            log.warning("audio was not finite, skipping audio saving")
            return None

        for _ in range(self.n_iters):
            _, phase = librosa.magphase(librosa.stft(signal, n_fft=self.n_fft))
            complex_spec = magnitudes * phase
            signal = librosa.istft(complex_spec)
        return signal
