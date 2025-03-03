%均值可以视为直流分量，0HZ
%ff是傅里叶系数，其长度与原信号相等，幅值表示频率所占的比重
%如采样频率fs=500 hz，可采信号最大频率为250 hz，若信号长度为N，则
%第n个系数代表的频率fn=n*fs/N，n<=N/2
function A=periodNum(B)
l=length(B);
ff=abs(fft(B-mean(B)));  %http://www.ilovematlab.cn/thread-301348-1-1.html；减去均值，约等于去掉直流分量
% [b1 b2]=max(ff);         %b1是幅值最大值，b2是对应的位置。幅值最大即可视为在原信号中能量最大，是在l中主信号的频率，
%                          %即脉搏波在l中的个数
% A=l/b2;                  %http://wenku.baidu.com/view/c51fab2758fb770bf78a5551.html；l的长度除以脉搏波个数可得每个
%周期的长度
ff=ff(1:500);
extrMaxIndex = find(diff(sign(diff(ff)))==-2)+1;

[a1 a2]=max(ff);


c=round(a2/2);
z=extrMaxIndex-c;
[ind,value]=find(z==min(abs(z)));
if value<3
    
    b1=ff(extrMaxIndex(ind));
    b2=extrMaxIndex(ind);
    if a1-b1<a1/3
        A=l/b2; 
    else
        A=l/a2; 
    end
else
    A=l/a2; 
end




