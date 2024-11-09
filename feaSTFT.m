function [ fea ] = feaSTFT( X )

if ~isempty(X)
    [fea,F,T,P]=spectrogram(X,8,4,19,1);
    %fea=abs(spectrogram(X,8,4));
    fea=abs(fea);
    fea=fea(:)';
else
    fea=zeros(1,150);
end
% figure;
% plot(X);
% figure;
% plot(fea);
end

