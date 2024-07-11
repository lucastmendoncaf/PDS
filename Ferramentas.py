import numpy as np
import wave
import scipy
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import scipy.signal


def Dados (Musicona):
    """

    Parameters:

        s_rate: Return the sample rate ex: 16 KHz
        n_samples: The amount of points in the audio file
        time: Total time of the audio
        signal: the signal of the file in int16 format
    """
    wav_obj = wave.open(Musicona, 'rb') #Read the wav file
    s_rate = wav_obj.getframerate() #Get the sample rate of the file
    n_samples = wav_obj.getnframes() #Get the amount of samples in the file
    t_audio = n_samples/s_rate # calculate the Total time of the audio file
    signal_wave = wav_obj.readframes(n_samples) # get the frames of the audio file
    signal = np.frombuffer(signal_wave, dtype=np.int16) #transform the signal_wave in a int16 

    #n_channels = wav_obj.getnchannels()
    
    return(s_rate, n_samples, t_audio, signal )
    #s_rate , n_samples, t_audio, signal = Dados(Pedrada)

def kowalski (Pedrada):
    
    s_rate , _, _, signal = Dados(Pedrada)

    FFT = abs(scipy.fft.fft(signal))
    freqs = scipy.fftpack.fftfreq(len(FFT), (1.0/s_rate))

    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])                                                          
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

def Amplitude (Pedrada):

    s_rate , n_samples, t_audio , signal = Dados(Pedrada)
    times = np.linspace(0, n_samples/s_rate, num=n_samples)

    plt.figure(figsize=(15, 5))
    plt.plot(times, signal)
    plt.title('Channel')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.show()
    
def Espectro (Pedrada):

    s_rate , _, t_audio, signal = Dados(Pedrada)

    plt.figure(figsize=(15, 5))
    plt.specgram(signal, Fs=s_rate, vmin=-20, vmax=50)
    plt.title('Left Channel')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.xlim(0, t_audio)
    plt.colorbar()
    plt.show()

def Filtrinho(Pedrada):

    s_rate , n_samples, t_audio, signal = Dados(Pedrada)

    times = np.linspace(0, n_samples/s_rate, num=n_samples)

    cutoff = 1000      # desired cutoff frequency of the filter, Hz , 
    nyq = 0.5 * s_rate  # Nyquist Frequency
    order = 9       # sin wave can be approx represented as quadratic

    def butter_lowpass_filter(signal, cutoff, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = scipy.signal.filtfilt(b, a, signal) #Signal filtered
        
        return y
    
    y = butter_lowpass_filter(signal, cutoff, order)
    
    scipy.io.wavfile.write('abc2.wav', s_rate, y.astype(np.int16))

def Ruido():

    s_rate = 2000
    x = 60
    t = np.arange(0, 1 , 1/s_rate)  
    signal = (np.cos(2*np.pi*50*t) + np.cos(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*150*t) + 0.5*np.sin(2*np.pi*300*t))

    #signal += signal.astype(np.int16)

    FFT = abs(scipy.fft.fft(signal))
    freqs = scipy.fftpack.fftfreq(len(FFT), (1.0/s_rate))
    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])
    #plt.xlim(0 , 4000)                                                          
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    times = np.linspace(0, 2000/s_rate, num=2000)

    plt.figure(figsize=(15, 5))
    plt.plot(times, signal)
    plt.title('Channel')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, 1)
    plt.show()


    #scipy.io.wavfile.write('Noise1_1.wav', s_rate, signal.astype(np.int16))
    #scipy.io.wavfile.write('Noise1.wav', s_rate, signal.astype(np.int16))

    print("arroz")

def Filtro_Manual(Pedrada):

    s_rate , n_samples, t_audio, signal = Dados(Pedrada)
    # Normalize the cutoff frequency
    
    cutoff = 1000      # desired cutoff frequency of the filter, Hz , 
    nyq = 0.5 * s_rate  # Nyquist Frequency
    order = 9       # sin wave can be approx represented as quadratic

    # Calculate the poles of the Butterworth filter
    poles = np.exp(1j * np.pi * (2 * np.arange(order) + 1 + order % 2) / (2 * order))

    #Normalization of the poles
    Cut_rad = 2 * np.pi * cutoff #Cutoff in radians
    poles = poles*Cut_rad

    # Calculating the transfer function coefficients
    B = np.prod(-poles)
    A = np.poly(poles).real

    #Bilinear transform
    a , b = scipy.signal.bilinear(B, A, s_rate)

    # Apply the filter to the signal
    filtered_signal = np.convolve(signal, b / a[0])

    FFT = abs(scipy.fft.fft(filtered_signal))
    freqs = scipy.fftpack.fftfreq(len(FFT), (1.0/s_rate))

    plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])                                                          
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

    return filtered_signal







