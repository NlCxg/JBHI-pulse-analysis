function [ fea ] = feaSTFT( X )
%UNTITLED7 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
if ~isempty(X)
    [fea,F,T,P]=spectrogram(X,8,4,19,2);
    % 19 ��ʱ����10
    % 10 ��ʱ����6
    %fea=abs(spectrogram(X,8,4));
    fea=abs(fea);%ÿһ�а���һ�����ھֲ�ʱ���Ƶ�ʳɷֹ��ƣ�ʱ���������ӣ�Ƶ���������ӡ�
    fea=fea(:)';
else
    fea=zeros(1,150);
end
% figure;
% plot(X);
% figure;
% plot(fea);
end

