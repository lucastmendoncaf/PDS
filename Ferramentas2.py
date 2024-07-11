import numpy as np
import scipy.fft
import scipy.fftpack
import scipy.signal
import matplotlib.pyplot as plt
import wave

def Dados(Musicona):
    wav_obj = wave.open(Musicona, 'rb')
    s_rate = wav_obj.getframerate()
    n_samples = wav_obj.getnframes()
    t_audio = n_samples / s_rate
    signal_wave = wav_obj.readframes(n_samples)
    signal = np.frombuffer(signal_wave, dtype=np.int16)
    return s_rate, n_samples, t_audio, signal

def kowalski(Pedrada):
    s_rate, _, _, signal = Dados(Pedrada)
    FFT = np.abs(scipy.fft.fft(signal))
    freqs = scipy.fftpack.fftfreq(len(FFT), 1.0 / s_rate)
    return freqs, FFT

def Amplitude(Pedrada):
    s_rate, n_samples, t_audio, signal = Dados(Pedrada)
    times = np.linspace(0, n_samples / s_rate, num=n_samples)
    return times, signal

def Espectro(Pedrada):
    s_rate, _, t_audio, signal = Dados(Pedrada)
    return signal, s_rate, t_audio

def Filtrinho(Pedrada, cutoff=1000, order=9):
    s_rate, n_samples, t_audio, signal = Dados(Pedrada)
    nyq = 0.5 * s_rate
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = scipy.signal.filtfilt(b, a, signal)
    return y, b, a

def plot_time_domain(times, signal):
    plt.figure(figsize=(15, 5))
    plt.plot(times, signal)
    plt.title('Sinal de Áudio no Domínio do Tempo')
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo (s)')
    plt.show()

def plot_spectrum(freqs, FFT):
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.plot(freqs[:len(freqs) // 2], np.abs(FFT)[:len(freqs) // 2])
    plt.title('Espectro de Amplitude')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequência (Hz)')
    
    plt.subplot(2, 1, 2)
    plt.plot(freqs[:len(freqs) // 2], np.angle(FFT)[:len(freqs) // 2])
    plt.title('Espectro de Fase')
    plt.ylabel('Fase (radianos)')
    plt.xlabel('Frequência (Hz)')
    plt.show()

def plot_transfer_function(b, a, s_rate):
    w, h = scipy.signal.freqz(b, a, worN=8000)
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    plt.plot(0.5 * s_rate * w / np.pi, np.abs(h))
    plt.title('Função de Transferência do Filtro')
    plt.ylabel('Ganho')
    plt.xlabel('Frequência (Hz)')
    
    plt.subplot(2, 1, 2)
    plt.plot(0.5 * s_rate * w / np.pi, np.angle(h))
    plt.title('Espectro do Filtro')
    plt.ylabel('Fase (radianos)')
    plt.xlabel('Frequência (Hz)')
    plt.show()

def plot_poles_zeros(b, a):
    z, p, k = scipy.signal.tf2zpk(b, a)
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(z), np.imag(z), s=50, marker='o', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), s=50, marker='x', label='Polos')
    plt.axvline(0, color='k', lw=1)
    plt.axhline(0, color='k', lw=1)
    plt.title('Polos e Zeros da Função de Transferência')
    plt.xlabel('Real')
    plt.ylabel('Imaginário')
    plt.grid()
    plt.legend()
    plt.show()

def plot_filtered_spectrum(signal_filtered, s_rate):
    FFT_filtered = np.abs(scipy.fft.fft(signal_filtered))
    freqs_filtered = scipy.fftpack.fftfreq(len(FFT_filtered), 1.0 / s_rate)
    plot_spectrum(freqs_filtered, FFT_filtered)

def plot_filtered_time_domain(times, signal_filtered):
    plt.figure(figsize=(15, 5))
    plt.plot(times, signal_filtered)
    plt.title('Sinal de Áudio no Domínio do Tempo Após Processamento')
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo (s)')
    plt.show()


