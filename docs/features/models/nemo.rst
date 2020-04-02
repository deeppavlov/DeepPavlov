Speech recognition and synthesis (ASR and TTS)
==============================================

DeepPavlov contains models for automatic speech recognition (ASR) and text synthesis (TTS) based on pre-build modules
from `NeMo <https://nvidia.github.io/NeMo/index.html>`__ (v0.9.0) - NVIDIA toolkit for defining and building
Conversational AI applications. Named arguments for modules initialization are taken from the NeMo config file (please
do not confuse with the DeepPavlov config file that defines model pipeline).

Speech recognition
------------------

The ASR pipeline is based on Jasper: an CTC-based end-to-end model. The model transcripts speech samples without
any additional alignment information. :class:`~deeppavlov.models.nemo.asr.NeMoASR` contains following modules:

-  `AudioToMelSpectrogramPreprocessor <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_asr/nemo_asr/audio_preprocessing.py>`__ - uses arguments from `AudioToMelSpectrogramPreprocessor` section of the NeMo config file.
-  `JasperEncoder <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_asr/nemo_asr/jasper.py>`__ - uses arguments from `JasperEncoder` section of the NeMo config file. Needs pretrained checkpoint.
-  `JasperDecoderForCTC <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_asr/nemo_asr/jasper.py>`__ - uses arguments from `JasperDecoder` section of the NeMo config file. Needs pretrained checkpoint.
-  `GreedyCTCDecoder <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_asr/nemo_asr/greedy_ctc_decoder.py>`__ - doesn't use any arguments.
-  :class:`~deeppavlov.models.nemo.asr.AudioInferDataLayer` - uses arguments from `AudioToTextDataLayer` section of the NeMo config file.

Besides named arguments for the modules above NeMo config file for ASR should contain `labels` arg.

Speech synthesis
----------------

The TTS pipeline that creates human audible speech from text is based on Tacotron 2 and Waveglow models.
:class:`~deeppavlov.models.nemo.tts.NeMoTTS` contains following modules:

-  `TextEmbedding <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_tts/nemo_tts/tacotron2_modules.py>`__ - uses arguments from `TextEmbedding` section of the NeMo config file. Needs pretrained checkpoint.
-  `Tacotron2Encoder <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_tts/nemo_tts/tacotron2_modules.py>`__ - uses arguments from `Tacotron2Encoder` section of the NeMo config file. Needs pretrained checkpoint.
-  `Tacotron2DecoderInfer <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_tts/nemo_tts/tacotron2_modules.py>`__ - uses arguments from `Tacotron2Decoder` section of the NeMo config file. Needs pretrained checkpoint.
-  `Tacotron2Postnet <https://github.com/NVIDIA/NeMo/blob/v0.9.0/collections/nemo_tts/nemo_tts/tacotron2_modules.py>`__ - uses arguments from `Tacotron2Postnet` section of the NeMo config file. Needs pretrained checkpoint.
-  :class:`~deeppavlov.models.nemo.vocoder.WaveGlow` - uses arguments from `WaveGlowNM` section of the NeMo config file. Needs pretrained checkpoint.
-  :class:`~deeppavlov.models.nemo.vocoder.GriffinLim` - uses arguments from `GriffinLim` section of the NeMo config file.
-  :class:`~deeppavlov.models.nemo.tts.TextDataLayer` - uses arguments from `TranscriptDataLayer` section of the NeMo config file.

Besides named arguments for the modules above NeMo config file for ASR should contain `labels` and `sample_rate` args.

Audio encoding end decoding.
----------------------------

:class:`~deeppavlov.models.nemo.common.ascii_to_bytes_io` and :class:`~deeppavlov.models.nemo.common.bytes_io_to_ascii`
was added to the library to achieve uniformity at work with both text and audio data. Classes can be used to encode
binary data to ascii string and decode back.

Quck Start
----------

DeepPavlov has default :config:`config <nemo/asr_tts_faq.json>` that demonstrate how to use ASR and TTS with FAQ skill.
To train your own checkpoints see `ASR tutorial <https://nvidia.github.io/NeMo/asr/tutorial.html>`__
and `TTS tutorial <https://nvidia.github.io/NeMo/tts/tutorial.html>`__

.. warning::
    NeMo library v0.9.0 doesn't allow to infer batches longer than one without compatible NVIDIA GPU.


