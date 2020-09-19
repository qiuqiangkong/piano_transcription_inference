# Piano transcription inference

This toolbox is a piano transcription inference package that can be easily installed. Users can transcribe their favorite piano recordings to MIDI files after installation. To see how the piano transcription system is trained, please visit: https://github.com/bytedance/piano_transcription.

## Demos
Here is a demo of our piano transcription system: https://www.youtube.com/watch?v=5U-WL0QvKCg

## Installation
The piano transcription system is developed with Python 3.7 and PyTorch 1.4.0 (Should work with other versions, but not fully tested).
Install PyTorch (>=1.4) following https://pytorch.org/. Users should have **ffmpeg** installed to transcribe mp3 files.

```
pip install piano_transcription_inference
```

Then, installation finished! 

## Usage
```
python3 example.py --audio_path='resources/cut_liszt.mp3' --output_midi_path='cut_liszt.mid' --cuda
```

For example:
```
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

# Load audio
(audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

# Transcriptor
transcriptor = PianoTranscription(device='cuda')	# 'cuda' | 'cpu'

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'cut_liszt.mid')
```

## Visualization of piano transcription

**Demo.** Lang Lang: Franz Liszt - Love Dream (Liebestraum) [[audio]](resources/cut_liszt.mp3) [[transcribed_midi]](resources/cut_liszt.mid)

<img src="resources/cut_liszt.png">

## FAQs
If users met the problem of command line shows "Killed". This can be caused by users do not have sufficient memory.

## Applications

We have built a large-scale classical piano MIDI dataset https://github.com/bytedance/GiantMIDI-Piano using our piano transcription system.

## Cite
[1] High-resolution Piano Transcription with Pedals by Regressing Onsets and Offsets Times, [To appear], 2020