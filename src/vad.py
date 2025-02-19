from scipy import signal
from collections import deque
from abc import ABC, abstractmethod
from enum import Enum
import matplotlib.pyplot as plt
import webrtcvad
import numpy as np
import soundfile as sf
import librosa

def plot_hist(plots, bins=50):
    plt.hist(plots, bins=bins)
    plt.title('Distribution Histogram')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_signal(plots):
    x = [i for i in range(len(plots))]
    plt.plot(x, plots)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def transcription(file):
    from openai import OpenAI
    client = OpenAI()
    audio_file= open(file, "rb")
    transcription = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    return transcription.text

# convert float to int16 for PCM format
def float_to_int16(chunk):
    chunk = np.clip(chunk, -1.0, 1.0)
    chunk = (chunk * 32767).astype(np.int16)
    return chunk.tobytes()

"""
Calculate the baseline using webrtcvad
More info: https://github.com/wiseman/py-webrtcvad/blob/e283ca41df3a84b0e87fb1f5cb9b21580a286b09/cbits/webrtc/common_audio/vad/vad_core.c#L133

Args:
    audio: numpy array of audio samples
    sr: int, the sampling rate of the audio

Returns:
    float: boolean array of whether each audio chunk is speech or not
"""
def calculate_baseline(audio, sr):
    chunk_size = int(sr * 0.01)
    vad = webrtcvad.Vad(3)
    result = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        chunk = float_to_int16(chunk)
        if len(chunk) == chunk_size:
            # only 10ms, 20ms, or 30ms chunks are allowed in webrtcvad
            result.append(vad.is_speech(chunk, sr))
    return result

class RecordState(Enum):
    FILLER=1
    WORD=2
    NOISE=3

class State(ABC):
    def __init__(self, turn_detector):
        self.turn_detector = turn_detector
    @abstractmethod
    def detect(self):
        pass

class Ended(State):
    def detect(self, audio_chunk):
        if self.turn_detector.records[-1] == RecordState.WORD:
            # print("STATE CHANGED FROM ENDED TO SPEAKING")
            self.turn_detector.change_state(Speaking(self.turn_detector))
            return True
        # print("ENDED")
        return False

class Speaking(State):
    def silenced(self):
        return all(list(map(lambda x: x == RecordState.NOISE, list(self.turn_detector.records)[-9:])))

    def detect(self, audio_chunk):
        # use records array instead of calling it here directly
        if self.silenced():
            if self.turn_detector.records[-10] == RecordState.FILLER:
                # print("STATE CHANGED FROM SPEAKING TO PAUSED")
                self.turn_detector.change_state(Paused(self.turn_detector))
                return True
            # print("STATE CHANGED FROM SPEAKING TO ENDED")
            self.turn_detector.change_state(Ended(self.turn_detector)) # TODO: might need to reset ema
            return False
        # print("SPEAKING")
        return True

class Paused(State):
    def silenced(self):
        return all(list(map(lambda x: x == RecordState.NOISE, list(self.turn_detector.records)[-24:])))

    def detect(self, audio_chunk):
        if self.silenced() and self.turn_detector.records[-25] == RecordState.FILLER:
            # print("STATE CHANGED FROM PAUSED TO ENDED")
            self.turn_detector.change_state(Ended(self.turn_detector))
            return False
        if self.turn_detector.records[-1] == RecordState.WORD:
            # print("STATE CHANGED FROM PAUSED TO SPEAKING")
            self.turn_detector.change_state(Speaking(self.turn_detector))
        return True

class TurnDetector:
    def __init__(self, reference_std=1.0, threshold=0.5, alpha=0.1, sampling_rate=16000):
        """
        Initialize your turn detector here
        You probably want to track history / state between audio chunks.

        Args:
            reference_std: float, the standard deviation of noise
            threshold: float, the threshold for determining turn completion
            alpha: float, the smoothing factor for the EMA

        """
        # Your implementation here
        self.ema_std = reference_std
        self.last_ema_std = 0
        self.threshold = threshold
        self.alpha = alpha
        self.sampling_rate = sampling_rate
        self.state = Ended(self)
        self.pitches = deque([np.nan] * 10, maxlen=10)  # hold previous 10 samples' pitches = 1 second
        self.records = deque([]) # hold previous 20 samples' states: word, filler, noise
        # TODO: put noise inside incase running into errors; don't put it there for debugging to match time

    def update_pitches(self, audio_chunk):
        self.pitches.append(self.extract_pitch(audio_chunk))

    def update_records(self, audio_chunk):
        if self.detect_filler():
            self.records.append(RecordState.FILLER)
        elif self.ema(audio_chunk):
            self.records.append(RecordState.WORD)
        else:
            self.records.append(RecordState.NOISE)
        
    def basic_vad(self, audio_chunk, power_threshold=0.1):
        """
        Basic VAD using power threshold
        Returns True if speech detected in chunk
        """
        return np.mean(audio_chunk ** 2) > power_threshold

    def filter_audio(self, audio_chunk):
        """
        Filter the audio chunk using a bandpass filter
        """
        sos = signal.butter(4, [300, 1500], 'bandpass', fs=self.sampling_rate, output='sos')
        filtered = signal.sosfilt(sos, audio_chunk)
        return filtered

    def ema(self, audio_chunk):
        """
        EMA implementation

        Args:
            audio_chunk: numpy array of audio samples

        Returns:
            bool: True if the EMA std deviation exceeds the threshold, False otherwise
        """
        filtered = self.filter_audio(audio_chunk)
        std = np.std(filtered)
        deviation = abs(std)  # Absolute deviation from mean (assumed ~0)
        ema_std = self.alpha * deviation + (1 - self.alpha) * self.ema_std  # EMA update
        # run exponential moving average on it
        if abs(ema_std - self.ema_std) > self.threshold:
            self.last_ema_std = self.ema_std
            self.ema_std = ema_std  # Optional: Adapt reference dynamically
            return True
        return False

    def extract_pitch(self, audio_chunk):
        f0, voiced_flag, _ = librosa.pyin(audio_chunk, fmin=50, fmax=300)
        # nanmean over mean because librosa returns a lot of NaN
        return np.nanmean(f0)
    
    def detect_filler(self, leniency=7, window_size=5): # 7 is good for all except korean_paused
        first = self.pitches[-window_size]
        mean = np.nanmean(list(self.pitches)[-window_size:])
        last = self.pitches[-1]
        paused = True # paused is only true when last < mean < first and each diff is within leniency
        for i in range(window_size):
            index = i - window_size
            if np.isnan(self.pitches[index]) or abs(self.pitches[index]-self.pitches[index-1]) > leniency:
                paused = False
                break
        return paused and last < mean < first

    def change_state(self, new_state):
        self.state = new_state
    
    def detect_turn_completion(self, audio_chunk):
        """
        Your improved turn detection implementation
        Should determine if speaker has completed their turn

        You may want to look at features outside of power.
        E.x zero crossing rate, pitch / frequency changes, or energy contour. 
        
        Args:
            audio_chunk: numpy array of audio samples
            sampling_rate: int, the sampling rate of the audio chunk
            
        Returns:
            bool: True if turn is complete, False otherwise
        """
        # Your implementation here
        # Consider:
        # - Mid-sentence pauses
        # - Filler words
        # - Language-agnostic features
        self.update_pitches(audio_chunk)
        self.update_records(audio_chunk)
        return self.state.detect(audio_chunk)

def main():
    """Example usage"""
    # Load and process a test file
    audio, sr = sf.read('../data/english_normal.wav')

    detector = TurnDetector(reference_std=0.000246, alpha=0.005, threshold=0.0001, sampling_rate=sr)
    
    # use state-of-the-art to establish baseline
    # baseline = calculate_baseline(audio, sr)
    # plot_signal(baseline)

    # TODO: need to implement the state transition
    # TODO: need a lookahead buffer to catch first few samples before ema picks up
    # TODO: once conv has ended, it might need to reset the self.ema_std

    # Process audio in chunks
    chunk_size = int(sr * 0.1)
    plots = []
    prediction = []
    pitches = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) == chunk_size:
            # plots.append(chunk)
            # pitches.append(extract_pitch(chunk))
            # prediction.append(detector.detect_turn_completion(chunk))
            print(f"Time {i/sr:.2f}s: Turn complete? {not detector.detect_turn_completion(chunk)}")

    # print(f"Number of Samples: {len(plots)}")
    # print(transcription())
    # plot_signal(plots)
    # plot_signal(prediction)
    # plot_signal(pitches)
    # plot_signal(detect_pause(pitches))
    # plot_signal([rec.value for rec in detector.records])

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()