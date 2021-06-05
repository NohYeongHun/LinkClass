# 신호처리 (FFT and)
Signal Representation by Harmonic Sinusoids

FFT with Smapling Frequency

x[n] = 2cos(2*pi*60t) => f = 60 &nbsp; 60t => frequency(주기)
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from scipy.fftpack import fft, fftshift
from scipy.signal import spectrogram

Fs = 2**10 # Sampling frequency
T = 1/Fs # Sampling period ( or sampling interval)

N = 5000 # Total data points (signal length)

t = np.arange(0, N) * T # Time vector (time range)

k = np.arange(0, N) # vector from 0 to N-1
f = (Fs/N) *k # frequency range

x = 2*np.cos(2*np.pi*60*t) + np.random.randn(N) # 신호 + 랜덤 노이즈 
```
