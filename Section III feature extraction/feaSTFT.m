function [ fea ] = feaSTFT( X )
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
if ~isempty(X)
    [fea,F,T,P]=spectrogram(X,8,4,19,2);
    % 19 的时候是10
    % 10 的时候是6
    %fea=abs(spectrogram(X,8,4));
    fea=abs(fea);%每一列包含一个短期局部时间的频率成分估计，时间沿列增加，频率沿行增加。
    fea=fea(:)';
else
    fea=zeros(1,150);
end
% figure;
% plot(X);
% figure;
% plot(fea);
end

