import utmos
model = utmos.Score() # The model will be automatically downloaded and will automatically utilize the GPU if available.
print(model.calculate_wav_file('../../generated_files/p232_160_finally_0300.wav')) # -> Float
# or model.calculate_wav(wav, sample_rate)