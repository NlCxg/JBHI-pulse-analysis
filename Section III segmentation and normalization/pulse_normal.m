function A=pulse_normal(pulsedata,L,A)

tmp1=AmpNorm(pulsedata,A);
A=resamplewave(tmp1,L);
end
