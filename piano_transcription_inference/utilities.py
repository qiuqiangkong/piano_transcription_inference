import os
import numpy as np
from mido import MidiFile

from .piano_vad import note_detection_with_onset_offset_regress
from . import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


def write_events_to_midi(start_time, note_events, midi_path):
    """Write out note events to MIDI file.

    Args:
      start_time: float
      note_events: list of dict, e.g. [
        {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
        {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
        ...]
      midi_path: str
    """
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    
    # This configuration is the same as MIDIs in MAESTRO dataset
    ticks_per_beat = 384
    beats_per_second = 2
    ticks_per_second = ticks_per_beat * beats_per_second
    microseconds_per_beat = int(1e6 // beats_per_second)

    midi_file = MidiFile()
    midi_file.ticks_per_beat = ticks_per_beat

    # Track 0
    track0 = MidiTrack()
    track0.append(MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    track0.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track0.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track0)

    # Track 1
    track1 = MidiTrack()
    
    # Message rolls of MIDI
    message_roll = []

    for note_event in note_events:
        # Onset
        message_roll.append({'midi_note': note_event['midi_note'], 
            'time': note_event['onset_time'], 'velocity': note_event['velocity']})

        # Offset
        message_roll.append({'midi_note': note_event['midi_note'], 
            'time': note_event['offset_time'], 'velocity': 0})

    # Sort MIDI messages by time
    message_roll.sort(key=lambda note_event: note_event['time'])

    previous_ticks = 0
    for message in message_roll:
        this_ticks = int((message['time'] - start_time) * ticks_per_second)
        if this_ticks >= 0:
            diff_ticks = this_ticks - previous_ticks
            previous_ticks = this_ticks
            track1.append(Message('note_on', note=message['midi_note'], velocity=message['velocity'], time=diff_ticks))
    track1.append(MetaMessage('end_of_track', time=1))
    midi_file.tracks.append(track1)

    midi_file.save(midi_path)


class RegressionPostProcessor(object):
    def __init__(self, frames_per_second, classes_num, onset_threshold, 
        offset_threshold, frame_threshold):
        """Postprocess the output of a transription model to MIDI events.

        Args:
          frames_per_second: int
          classes_num: int
          onset_threshold: float
          offset_threshold: float
          frame_threshold: float
        """
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.begin_note = config.begin_note
        self.velocity_scale = config.velocity_scale

    def output_dict_to_midi_events(self, output_dict):
        """Postprocess the output of a transription model to MIDI events.

        Args:
          output_dict: dict, {
            'reg_onset_output': (frames_num, classes_num), 
            'reg_offset_output': (frames_num, classes_num), 
            'frame_output': (frames_num, classes_num), 
            'velocity_output': (frames_num, classes_num)
          }

        Returns:
          est_note_events: e.g., [
            {'midi_note': 34, 'onset_time': 32.837551682293416, 'offset_time': 35.77, 'velocity': 101}, 
            {'midi_note': 34, 'onset_time': 37.37115609429777, 'offset_time': 39.93, 'velocity': 103}
            ...]
        """

        # Calculate and append onset, onset shift, offset, offset shift to output_dict
        output_dict = self.get_onset_roll_from_regression(output_dict)
        output_dict = self.get_offset_roll_from_regression(output_dict)

        # Detect piano notes from output_dict
        (est_on_off_vels, est_midi_notes) = self.output_dict_to_midi_notes(output_dict)

        # Reformat piano notes to MIDI events
        est_note_events = self.notes_to_midi_events(
            est_on_off_vels, est_midi_notes)

        return est_note_events

    def get_onset_roll_from_regression(self, output_dict):
        """Get onset roll and onset shift from the regression result.

        Args:
          output_dict: dict, {'reg_onset_output': (frames_num, classes_num), ...}

        Returns:
          output_dict: dict, append 'onset_output': (frames_num, classes_num), 
            'onset_shift_output': (frames_num, classes_num) to output_dict
        """
        reg_onset_output = output_dict['reg_onset_output']
        onset_output = np.zeros_like(reg_onset_output)
        onset_shift_output = np.zeros_like(reg_onset_output)
        (frames_num, classes_num) = reg_onset_output.shape
        
        for k in range(classes_num):
            x = reg_onset_output[:, k]
            for n in range(2, frames_num - 2):
                if x[n] > x[n - 1] > x[n - 2] \
                    and x[n] > x[n + 1] > x[n + 2] \
                    and x[n] > self.onset_threshold:

                    onset_output[n, k] = 1

                    """See Q. Kong, et al., High resolution piano transcription by 
                    regressing onset and offset time stamps, Section 3.4 for deduction"""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    onset_shift_output[n, k] = shift

        output_dict['onset_output'] = onset_output
        output_dict['onset_shift_output'] = onset_shift_output
        return output_dict


    def get_offset_roll_from_regression(self, output_dict):
        """Get onset roll and onset shift from the regression result.

        Args:
          output_dict: dict, {'reg_onset_output': (frames_num, classes_num), ...}

        Returns:
          output_dict: dict, append 'onset_output': (frames_num, classes_num), 
            'offset_shift_output': (frames_num, classes_num) to output_dict
        """
        reg_offset_output = output_dict['reg_offset_output']
        offset_output = np.zeros_like(reg_offset_output)
        offset_shift_output = np.zeros_like(reg_offset_output)
        (frames_num, classes_num) = reg_offset_output.shape
        
        for k in range(classes_num):
            x = reg_offset_output[:, k]
            for n in range(4, frames_num - 4):
                if x[n] > x[n - 1] > x[n - 2] > x[n - 3] > x[n - 4] \
                    and x[n] > x[n + 1] > x[n + 2] > x[n + 3] > x[n + 4] \
                    and x[n] > self.offset_threshold:

                    offset_output[n, k] = 1

                    """See Q. Kong, et al., High resolution piano transcription by 
                    regressing onset and offset time stamps, Section 3.4 for deduction"""
                    if x[n - 1] > x[n + 1]:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
                    else:
                        shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
                    offset_shift_output[n, k] = shift

        output_dict['offset_output'] = offset_output
        output_dict['offset_shift_output'] = offset_shift_output
        return output_dict

    def output_dict_to_midi_notes(self, output_dict):
        """Postprocess output_dict to piano notes.

        Args:
          output_dict: dict, e.g. {
            'onset_output': (frames_num, classes_num),
            'onset_shift_output': (frames_num, classes_num),
            'offset_output': (frames_num, classes_num),
            'offset_shift_output': (frames_num, classes_num),
            'frame_output': (frames_num, classes_num),
            'onset_output': (frames_num, classes_num),
            ...}

        Returns:
          est_pairs: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700,  0.7932],
             [37.3712, 39.9300,  0.8058],
             ...]
          est_midi_notes: (notes,), e.g., [34, 34, 35, 35, ...]
        """
        est_tuples = []
        est_midi_notes = []
        classes_num = output_dict['frame_output'].shape[-1]
 
        for piano_note in range(classes_num):
            """Detect piano notes"""
            est_tuples_per_note = note_detection_with_onset_offset_regress(
                frame_output=output_dict['frame_output'][:, piano_note], 
                onset_output=output_dict['onset_output'][:, piano_note], 
                onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
                offset_output=output_dict['offset_output'][:, piano_note], 
                offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
                velocity_output=output_dict['velocity_output'][:, piano_note], 
                frame_threshold=self.frame_threshold)
            
            est_tuples += est_tuples_per_note
            est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

        est_tuples = np.array(est_tuples)   # (notes, 5)
        est_midi_notes = np.array(est_midi_notes) # (notes,)

        onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
        offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
        velocities = est_tuples[:, 4]
        est_on_off_vel = np.stack((onset_times, offset_times, velocities), axis=-1)
        """(notes, 3), the three columns are onset_times, offset_times and velocity."""

        return est_on_off_vel, est_midi_notes

    def notes_to_midi_events(self, est_on_off_vels, est_midi_notes):
        """Reformat detected notes to midi events.

        Args:
          est_on_off_vels: (notes, 3), the three columns are onset_times, 
            offset_times and velocity. E.g.
            [[32.8376, 35.7700,  0.7932],
             [37.3712, 39.9300,  0.8058],
             ...]
          est_midi_notes: (notes,), e.g., [34, 34, 35, 35, ...]
        """
        midi_events = []
        for i in range(est_on_off_vels.shape[0]):
            velocity = int(est_on_off_vels[i][2] * self.velocity_scale)

            midi_events.append({'midi_note': est_midi_notes[i], 
                'onset_time': est_on_off_vels[i][0], 
                'offset_time': est_on_off_vels[i][1], 
                'velocity': velocity})

        return midi_events
