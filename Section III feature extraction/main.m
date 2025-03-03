clc;clear
pulse_data = load('singel_pulse_cycle');
singel_pulse_cycle = pulse_data.data_cycle;

load('Gabor_64.mat');
fea_gabor =feaGabor(singel_pulse_cycle,dict,9);fea_gabor = fea_gabor(1,1:384);
fea_stft =feaSTFT(singel_pulse_cycle);
fea_dfs =feaFourier_6(singel_pulse_cycle);




