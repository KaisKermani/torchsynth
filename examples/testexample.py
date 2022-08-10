import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import numpy as np
import torch
from torch import tensor
from torchsynth.config import SynthConfig
from torchsynth.module import (
    ADSR,
    VCA,
    ControlRateUpsample,
    MonophonicKeyboard,
    Noise,
    SineVCO,
    FmVCO,
)
from torchsynth.parameter import ModuleParameterRange


def time_plot(signal, sample_rate=44100, show=True):
    t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    if show:
        plt.show()


def stft_plot(signal, sample_rate=44100):
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
    plt.show()


# Run examples on GPU if available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# ## Globals
# We'll generate 2 sounds at once, 4 seconds each
synthconfig = SynthConfig(
    batch_size=2, reproducible=False, sample_rate=44100, buffer_size_seconds=4.0
)
# For a few examples, we'll only generate one sound
synthconfig1 = SynthConfig(
    batch_size=1, reproducible=False, sample_rate=44100, buffer_size_seconds=4.0
)

# Making the pitch modulation sinewave
sine_vco = SineVCO(
    tuning=tensor([0.0, 0.0]),
    synthconfig=synthconfig,
    mod_depth=tensor([-1.0, .2]),
).to(device)
mod_sig = sine_vco(freq=tensor([1.0, 5.0], device=device))
# time_plot(mod_sig[0].detach().cpu())

vibratoed_sine = sine_vco(tensor([115.0, 69.0]), mod_signal=mod_sig)
# stft_plot(vibratoed_sine[0].detach().cpu().numpy())
stft_plot(vibratoed_sine[1].detach().cpu().numpy())
wave.write("../audio_files/test.wav", 44100, vibratoed_sine[1].detach().cpu().numpy())
