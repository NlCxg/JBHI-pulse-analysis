function B=m_denoise(A)
A=double(A);
t=wpdec(A,4,'sym8','shannon');
%[THR,SORH,KEEPAPP,CRIT] = ddencmp('den','wp',A);
 t=besttree(t);
THR=wthrmngr('wp1ddenoGBL','bal_sn',t);
%CRITΪ������KEEPAPPȡֵΪ1ʱ�����Ƶϵ����������ֵ����������֮�����Ƶϵ��������ֵ����
CRIT='shannon';
KEEPAPP=0;
SORH='h';
[B,tree,perf0,perfl2]=wpdencmp(A,SORH,4,'sym8',CRIT,THR,KEEPAPP);

%Ps=sum((A).^2);%signal power
%Pn=sum((B-A).^2);%noise power
% snr=10*log10(Ps/Pn)
% mse=Pn/length(A)
% plot(B(1:2500));
% hold on;
% plot(A(1:2500));