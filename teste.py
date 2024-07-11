import Ferramentas as F
import Ferramentas2 as F2

#Loads the Audio file, O = original, B = with butter funcion , N = Generated Noise
Audio_O = 'som1s2024.wav'
Audio_B = 'Audio_Butter.wav' 
Audio_N = 'Noise1_1.wav'

#F.Filtrinho(Pedrada)

#F.kowalski(Audio_B)
#F.Amplitude(Audio_B)
#F.Espectro(Audio_B)
#F.Graficuzinho(Audio_B)

#F.Filtro_Manual(Audio_O)

arquivo_audio = 'som1s2024.wav'
times, signal = F2.Amplitude(arquivo_audio)
F2.plot_time_domain(times, signal)

freqs, FFT = F2.kowalski(arquivo_audio)
F2.plot_spectrum(freqs, FFT)

y, b, a = F2.Filtrinho(arquivo_audio)
F2.plot_transfer_function(b, a, times[-1])

F2.plot_poles_zeros(b, a)

F2.plot_filtered_spectrum(y, times[-1])

F2.plot_filtered_time_domain(times, y)


#F.Ruido()


#print(a)

