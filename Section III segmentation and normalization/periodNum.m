%��ֵ������Ϊֱ��������0HZ
%ff�Ǹ���Ҷϵ�����䳤����ԭ�ź���ȣ���ֵ��ʾƵ����ռ�ı���
%�����Ƶ��fs=500 hz���ɲ��ź����Ƶ��Ϊ250 hz�����źų���ΪN����
%��n��ϵ�������Ƶ��fn=n*fs/N��n<=N/2
function A=periodNum(B)
l=length(B);
ff=abs(fft(B-mean(B)));  %http://www.ilovematlab.cn/thread-301348-1-1.html����ȥ��ֵ��Լ����ȥ��ֱ������
% [b1 b2]=max(ff);         %b1�Ƿ�ֵ���ֵ��b2�Ƕ�Ӧ��λ�á���ֵ��󼴿���Ϊ��ԭ�ź��������������l�����źŵ�Ƶ�ʣ�
%                          %����������l�еĸ���
% A=l/b2;                  %http://wenku.baidu.com/view/c51fab2758fb770bf78a5551.html��l�ĳ��ȳ��������������ɵ�ÿ��
%���ڵĳ���
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




