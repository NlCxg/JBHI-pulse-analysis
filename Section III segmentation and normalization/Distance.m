%PeakDetection


function [L] = Distance(s)



x1=min(s);             %��ֵ
x2=max(s);
a=(s-x1)*50/(x2-x1);

la=length(s);
x=[1:la];                    %��ֵ�㣬����
xx=[1:(la-1)/49:la];         %��Ҫ����ĵ�,����50
a=interp1(x,a',xx);

x=[1:50];
L=sum(abs(x-a));

end
