function a=resamplewave(b,L)
lb=length(b);
x=[1:lb];                    %插值点
xx=[1:(lb-1)/(L-1):lb];         %需要计算的点,
a=interp1(x,b',xx);
end
