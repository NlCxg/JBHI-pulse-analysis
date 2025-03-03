clc;clear
pulse_data = load('pulse_data');
pulse_data = pulse_data.pulse_data;
dim = 200;

[pulse_cycles,pp,fp,B]=BaselineAd_diff(pulse_data');
if ~isempty(pp)
    if ~isempty(B)
        P_denoise=B(200:end-200);
        pulse_cycles=outlier_check(pulse_cycles,fp,dim,1);
        data_cycle=mean(pulse_cycles,1);     
    else
        P_denoise=[];
        data_cycle=[];
        tt_cycle=[];
        amplitude = [];
    end
end         

pulse_cycles;     % segmented and normaliz data
