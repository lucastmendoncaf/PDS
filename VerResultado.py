import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
from matplotlib import pyplot as plt
import Ferramentas as F
import wave

Audio_O = 'som1s2024.wav'
#y = 'abc0.wav'
s_rate , _, _, signal = F.Dados(Audio_O)

# Example usage
cutoff_freq = 1000  # Cutoff frequency in Hz
order = 9  # Filter order

y = F.butterworth_lowpass_filter(signal, cutoff_freq, s_rate, order)

wav_obj = wave.open(y, 'rb')
s_rate = wav_obj.getnframes()
print(signal)
print(s_rate)



#scipy.io.wavfile.write('abc0.wav', s_rate, y.astype(np.int16))

#F.Amplitude(y)