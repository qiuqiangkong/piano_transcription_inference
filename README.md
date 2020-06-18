# Piano transcription inference

This toolbox provide easy to use command for piano transcription inference.

# Installation
Install PyTorch (>=1.0) following https://pytorch.org/

```
$ python3 setup.py install
```

# Usage
```
python3 example.py --audio_path='resources/cut_liszt.mp3' --output_midi_path='cut_liszt.mid' --cuda
```

For example:
```
# Load audio
(audio, _) = librosa.core.load('resources/cut_liszt.mp3', sr=sample_rate, mono=True)

# Transcriptor
transcriptor = PianoTranscription(device=device)

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'cut_liszt.mid')
```

# Cite
[1] Q. Kong, et al., High resolution piano transcription by regressing onset and offset time stamps, [To appear], 2020