Speech recognition and synthesis (ASR and TTS)
==============================================

DeepPavlov contains models for automatic speech recognition (ASR) and text synthesis (TTS) based on pre-build modules
from `NeMo <https://nvidia.github.io/NeMo/index.html>`__ (v0.10.0) - NVIDIA toolkit for defining and building
Conversational AI applications. Named arguments for modules initialization are taken from the NeMo config file (please
do not confuse with the DeepPavlov config file that defines model pipeline).

Speech recognition
------------------

The ASR pipeline is based on Jasper: an CTC-based end-to-end model. The model transcripts speech samples without
any additional alignment information. :class:`~deeppavlov.models.nemo.asr.NeMoASR` contains following modules:

-  `AudioToMelSpectrogramPreprocessor <https://github.com/NVIDIA/NeMo/blob/v0.10.0/nemo/collections/asr/audio_preprocessing.py>`_ - uses arguments from ``AudioToMelSpectrogramPreprocessor`` section of the NeMo config file.
-  `JasperEncoder <https://nvidia.github.io/NeMo/collections/nemo_asr.html#nemo.collections.asr.jasper.JasperEncoder>`__ - uses arguments from ``JasperEncoder`` section of the NeMo config file. Needs pretrained checkpoint.
-  `JasperDecoderForCTC <https://nvidia.github.io/NeMo/collections/nemo_asr.html#nemo.collections.asr.jasper.JasperDecoderForCTC>`__ - uses arguments from ``JasperDecoder`` section of the NeMo config file. Needs pretrained checkpoint.
-  `GreedyCTCDecoder <https://github.com/NVIDIA/NeMo/blob/v0.10.0/nemo/collections/asr/greedy_ctc_decoder.py>`__ - doesn't use any arguments.
-  :class:`~deeppavlov.models.nemo.asr.AudioInferDataLayer` - uses arguments from ``AudioToTextDataLayer`` section of the NeMo config file.

NeMo config file for ASR should contain ``labels`` argument besides named arguments for the modules above. ``labels`` is
a list of characters that can be output by the ASR model used in model training.

Speech synthesis
----------------

The TTS pipeline that creates human audible speech from text is based on Tacotron 2 and Waveglow models.
:class:`~deeppavlov.models.nemo.tts.NeMoTTS` contains following modules:

-  `TextEmbedding <https://nvidia.github.io/NeMo/collections/nemo_tts.html#nemo.collections.tts.tacotron2_modules.TextEmbedding>`__ - uses arguments from ``TextEmbedding`` section of the NeMo config file. Needs pretrained checkpoint.
-  `Tacotron2Encoder <https://nvidia.github.io/NeMo/collections/nemo_tts.html#nemo.collections.tts.tacotron2_modules.Tacotron2Encoder>`__ - uses arguments from ``Tacotron2Encoder`` section of the NeMo config file. Needs pretrained checkpoint.
-  `Tacotron2DecoderInfer <https://nvidia.github.io/NeMo/collections/nemo_tts.html#nemo.collections.tts.tacotron2_modules.Tacotron2Decoder>`__ - uses arguments from ``Tacotron2Decoder`` section of the NeMo config file. Needs pretrained checkpoint.
-  `Tacotron2Postnet <https://nvidia.github.io/NeMo/collections/nemo_tts.html#nemo.collections.tts.tacotron2_modules.Tacotron2Postnet>`__ - uses arguments from ``Tacotron2Postnet`` section of the NeMo config file. Needs pretrained checkpoint.
-  :class:`~deeppavlov.models.nemo.vocoder.WaveGlow` - uses arguments from ``WaveGlowNM`` section of the NeMo config file. Needs pretrained checkpoint.
-  :class:`~deeppavlov.models.nemo.vocoder.GriffinLim` - uses arguments from ``GriffinLim`` section of the NeMo config file.
-  :class:`~deeppavlov.models.nemo.tts.TextDataLayer` - uses arguments from ``TranscriptDataLayer`` section of the NeMo config file.

NeMo config file for TTS should contain ``labels`` and ``sample_rate`` args besides named arguments for the modules
above. ``labels`` is a list of characters used in TTS model training.

Audio encoding end decoding.
----------------------------

:func:`~deeppavlov.models.nemo.common.ascii_to_bytes_io` and :func:`~deeppavlov.models.nemo.common.bytes_io_to_ascii`
was added to the library to achieve uniformity at work with both text and audio data. Components can be used to encode
binary data to ascii string and decode back.

Quck Start
----------

Preparation
~~~~~~~~~~~

Install requirements and download model files.

.. code:: bash

    python -m deeppavlov install asr_tts
    python -m deeppavlov download asr_tts

Examples below use `sounddevice <https://python-sounddevice.readthedocs.io/en/0.3.15/index.html>`_ library. Install
it with ``pip install sounddevice==0.3.15``. You may need to install ``libportaudio2`` package with
``sudo apt-get install libportaudio2`` to make ``sounddevice`` work.

.. note::
    ASR reads and TTS generates single channel WAV files. Files transferred to ASR are resampled to the frequency
    specified in the NeMo config file (16 kHz for models from DeepPavlov configs).

Speech recognition
~~~~~~~~~~~~~~~~~~

DeepPavlov :config:`asr <nemo/asr.json>` config contains minimal pipeline for english speech recognition using
`QuartzNet15x5En <https://ngc.nvidia.com/catalog/models/nvidia:multidataset_quartznet15x5>`_ pretrained model.
To record speech on your computer and print transcription run following script:

.. code:: python

    from io import BytesIO

    import sounddevice as sd
    from scipy.io.wavfile import write

    from deeppavlov import build_model, configs

    sr = 16000
    duration = 3

    print('Recording...')
    myrecording = sd.rec(duration*sr, samplerate=sr, channels=1)
    sd.wait()
    print('done')

    out = BytesIO()
    write(out, sr, myrecording)

    model = build_model(configs.nemo.asr)
    text_batch = model([out])

    print(text_batch[0])

Speech synthesis
~~~~~~~~~~~~~~~~

DeepPavlov :config:`tts <nemo/tts.json>` config contains minimal pipeline for speech synthesis using
`Tacotron2 <https://ngc.nvidia.com/catalog/models/nvidia:tacotron2_ljspeech>`_ and
`WaveGlow <https://ngc.nvidia.com/catalog/models/nvidia:waveglow_ljspeech>`_ pretrained models.
To generate audiofile and save it to hard drive run following script:

.. code:: python

    from deeppavlov import build_model, configs

    model = build_model(configs.nemo.tts)
    filepath_batch = model(['Hello world'], ['~/hello_world.wav'])

    print(f'Generated speech has successfully saved at {filepath_batch[0]}')

Speech to speech
~~~~~~~~~~~~~~~~

Previous examples assume files with speech to recognize and files to be generated are on the same system where the
DeepPavlov is running. DeepPavlov :config:`asr_tts <nemo/asr_tts.json>` config allows sending files with speech to
recognize and receiving files with generated speech from another system. This config is recognizes received speech and
re-sounds it.

Run ``asr_tts`` in REST Api mode:

.. code:: bash

    python -m deeppavlov riseapi asr_tts

This python script supposes that you already have file with speech to recognize. You can use code from speech
recognition example to record speech on your system. ``127.0.0.1`` should be replased by address of system where
DeepPavlov has started.

.. code:: python

    from base64 import encodebytes, decodebytes

    from requests import post

    with open('/path/to/wav/file/with/speech', 'rb') as fin:
        input_speech = fin.read()

    input_ascii = encodebytes(input_speech).decode('ascii')

    resp = post('http://127.0.0.1:5000/model', json={"speech_in_encoded": [input_ascii]})
    text, generated_speech_ascii = resp.json()[0]
    generated_speech = decodebytes(generated_speech_ascii.encode())

    with open('/path/where/to/save/generated/wav/file', 'wb') as fout:
        fout.write(generated_speech)

    print(f'Speech transcriptions is: {text}')

.. warning::
    NeMo library v0.10.0 doesn't allow to infer batches longer than one without compatible NVIDIA GPU.

Models training
---------------

To get your own pre-trained checkpoints for NeMo modules see `Speech recognition <https://nvidia.github.io/NeMo/asr/intro.html>`_
and `Speech Synthesis <https://nvidia.github.io/NeMo/tts/intro.html>`_ tutorials. Pre-trained models list could be found
`here <https://github.com/NVIDIA/NeMo/tree/v0.10.0#pre-trained-models>`_.