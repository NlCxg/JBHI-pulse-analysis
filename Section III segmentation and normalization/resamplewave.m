function a=resamplewave(b,L)
lb=length(b);
x=[1:lb];                    %��ֵ��
xx=[1:(lb-1)/(L-1):lb];         %��Ҫ����ĵ�,
a=interp1(x,b',xx);
end
