function [ pp,tt ] = periodseg_modified( peaks,lows,a,fh)
%UNTITLED7 此处显示有关此函数的摘要
%   此处显示详细说明
thr=0.25*fh;
if length(lows)>1
 for i=1:(length(lows)-1)
     tt{i}=a(lows(i):lows(i+1));
%     if abs(length(tt{i})-fh)>thr
%         tt{i}=[];
%     end
 end
else
    tt={};
end

if length(peaks)>3
  for i=1:(length(peaks)-1)
    pp{i}=a(peaks(i):peaks(i+1));
%     if abs(len gth(pp{i-1})-fh)>thr
%         pp{i-1}=[];
%     end
  end
else
    pp={};
end
pp(cellfun(@isempty,pp))=[];
tt(cellfun(@isempty,tt))=[];

end

