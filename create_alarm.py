from scipy.io.wavfile import write
import numpy as np

sample_rate = 44100
duration = 3

t = np.linspace(0, duration, int(sample_rate * duration), False)

tone1 = np.sin(2 * np.pi * 1200 * t)
tone2 = np.sin(2 * np.pi * 1800 * t)

signal = np.zeros_like(tone1)
segment = int(0.3 * sample_rate)

for i in range(0, len(t), segment):
    if (i // segment) % 2 == 0:
        signal[i:i+segment] = tone1[i:i+segment]
    else:
        signal[i:i+segment] = tone2[i:i+segment]

audio = np.int16(signal / np.max(np.abs(signal)) * 32767)

write("alarm.wav", sample_rate, audio)