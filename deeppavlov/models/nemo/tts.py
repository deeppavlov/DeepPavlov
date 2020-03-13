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

import os
from typing import Optional

import librosa
import nemo
import nemo_asr
import nemo_tts
import numpy as np
import torch
from ruamel.yaml import YAML
from scipy.io.wavfile import write

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable


def griffin_lim(magnitudes, n_iters=50, n_fft=1024):
    """
    Griffin-Lim algorithm to convert magnitude spectrograms to audio signals
    """
    phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
    if not np.isfinite(signal).all():
        print("WARNING: audio was not finite, skipping audio saving")
        return np.array([0])

    for _ in range(n_iters):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft))
        complex_spec = magnitudes * phase
        signal = librosa.istft(complex_spec)
    return signal


@register('nemo_tts')
class NeMoTTS(Component, Serializable):
    def __init__(self,
                 load_path: str,
                 tacotron2_conf: str,
                 tacotron2_ckpts_dir: str,
                 vocoder: str,
                 vocoder_config: str,
                 eval_dataset: str,
                 save_dir: str,
                 amp_opt_level: str = '01',
                 vocoder_model_load_dir: Optional[str] = None,
                 griffin_lim_mag_scale: float = 2048.0,
                 griffin_lim_power: float = 1.2,
                 waveglow_denoiser_strength: float = 0.0,
                 waveglow_sigma: float = 0.6,
                 **kwargs):
        super(NeMoTTS, self).__init__(save_path=None, load_path=load_path, **kwargs)
        placement = nemo.core.DeviceType.GPU if torch.cuda.is_available() else nemo.core.DeviceType.CPU
        neural_factory = nemo.core.NeuralModuleFactory(placement=placement)
        # TODO: find out why by default is true and becomes false if local_rank is not none
        use_cache = True

        yaml = YAML(typ="safe")
        with open(self.load_path / tacotron2_conf) as file:
            tacotron2_params = yaml.load(file)

        text_embedding = nemo_tts.TextEmbedding(
            len(tacotron2_params["labels"]) + 3,  # + 3 special chars
            **tacotron2_params["TextEmbedding"])
        t2_enc = nemo_tts.Tacotron2Encoder(**tacotron2_params["Tacotron2Encoder"])
        t2_dec = nemo_tts.Tacotron2DecoderInfer(
            **tacotron2_params["Tacotron2Decoder"])
        t2_postnet = nemo_tts.Tacotron2Postnet(
            **tacotron2_params["Tacotron2Postnet"])

        for step in (text_embedding, t2_enc, t2_dec, t2_postnet):
            step.restore_from(nemo.utils.get_checkpoint_from_dir([str(step)], self.load_path / tacotron2_ckpts_dir)[0])

        data_layer = nemo_asr.TranscriptDataLayer(
            path=eval_dataset,
            labels=tacotron2_params['labels'],
            batch_size=32,
            num_workers=1,
            load_audio=False,
            bos_id=len(tacotron2_params['labels']),
            eos_id=len(tacotron2_params['labels']) + 1,
            pad_id=len(tacotron2_params['labels']) + 2,
            shuffle=False
        )
        transcript, transcript_len = data_layer()

        transcript_embedded = text_embedding(char_phone=transcript)
        transcript_encoded = t2_enc(
            char_phone_embeddings=transcript_embedded,
            embedding_length=transcript_len)
        mel_decoder, gate, alignments, mel_len = t2_dec(
            char_phone_encoded=transcript_encoded,
            encoded_length=transcript_len)
        mel_postnet = t2_postnet(mel_input=mel_decoder)

        if vocoder == "waveglow":
            if not vocoder_config or not vocoder_model_load_dir:
                raise ValueError(
                    "Using waveglow as the vocoder requires the "
                    "vocoder_config and vocoder_model_load_dir")

            yaml = YAML(typ="safe")
            with open(vocoder_config) as file:
                waveglow_params = yaml.load(file)
            waveglow = nemo_tts.WaveGlowInferNM(sigma=waveglow_sigma, **waveglow_params["WaveGlowNM"])
            waveglow.restore_from(nemo.utils.get_checkpoint_from_dir([str(waveglow)], self.load_path / vocoder_model_load_dir)[0])
            if waveglow_denoiser_strength > 0:
                print("Setup denoiser")
                waveglow.setup_denoiser()
            audio_pred = waveglow(mel_spectrogram=mel_postnet)
            infer_tensors = [audio_pred, mel_len]

        elif vocoder == 'griffin-lim':
            infer_tensors = [mel_postnet, mel_len]

        else:
            raise ValueError(f"'{vocoder} vocoder does not supported.'")

        evaluated_tensors = neural_factory.infer(
            tensors=infer_tensors,
            cache=use_cache,
            offload_to_cpu=False
        )
        mel_len = evaluated_tensors[-1]
        print("Done Running Tacotron 2")
        filterbank = librosa.filters.mel(
            sr=tacotron2_params["sample_rate"],
            n_fft=tacotron2_params["n_fft"],
            n_mels=tacotron2_params["n_mels"],
            fmax=tacotron2_params["fmax"])

        if vocoder == "griffin-lim":
            print("Running Griffin-Lim")
            mel_spec = evaluated_tensors[0]
            for i, batch in enumerate(mel_spec):
                log_mel = batch.cpu().numpy().transpose(0, 2, 1)
                mel = np.exp(log_mel)
                magnitudes = np.dot(mel, filterbank) * griffin_lim_mag_scale
                for j, sample in enumerate(magnitudes):
                    sample = sample[:mel_len[i][j], :]
                    audio = griffin_lim(sample.T ** griffin_lim_power)
                    save_file = f"sample_{i*32+j}.wav"
                    if save_dir:
                        save_file = os.path.join(save_dir, save_file)
                    write(save_file, tacotron2_params["sample_rate"], audio)

        elif vocoder == "waveglow":
            print("Saving results to disk")
            for i, batch in enumerate(evaluated_tensors[0]):
                audio = batch.cpu().numpy()
                for j, sample in enumerate(audio):
                    sample_len = mel_len[i][j] * tacotron2_params["n_stride"]
                    sample = sample[:sample_len]
                    save_file = f"sample_{i*32+j}.wav"
                    if save_dir:
                        save_file = os.path.join(save_dir, save_file)
                    if waveglow_denoiser_strength > 0:
                        sample, spec = waveglow.denoise(
                            sample, strength=waveglow_denoiser_strength)
                    else:
                        spec, _ = librosa.core.magphase(librosa.core.stft(
                            sample, n_fft=waveglow_params["n_fft"]))
                    write(save_file, waveglow_params["sample_rate"], sample)

    def __call__(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    model = NeMoTTS('/data/nemo', 'tacotron2/tacotron2.yaml', 'tacotron2',
                    vocoder='waveglow',
                    vocoder_config='/home/ignatov/dev/NeMo/examples/tts/configs/waveglow.yaml',
                    vocoder_model_load_dir='/data/nemo/waveglow',
                    eval_dataset='/data/nemo/workdir/gen.json',
                    save_dir='/data/nemo/workdir')
