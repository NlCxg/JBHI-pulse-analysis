function [peaks,lows] = PeakDetection_Distance_Normalized( x,ff )
%UNTITLED3 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
%x=Normalized_amp(x);
N = length(x);
peaks = zeros(N,1);
%lows=zeros(N,1);

th = .5;
rng = floor(th/ff);

if(abs(max(x))>abs(min(x)))%max�϶�����min��x��û�и�ֵ
    for j = 1:N,
        %         index = max(j-rng,1):min(j+rng,N);
        if(j>rng & j<N-rng)
            index = j-rng:j+rng;
        elseif(j>rng)
            index = N-2*rng:N;
        else
            index = 1:2*rng;
        end
        
        if(max(x(index))==x(j))
            peaks(j) = 1;
        end
        %-----------------------ѡ�񴰿�����С��Ϊ���ȵ�-----------------------------
        if(min(x(index))==x(j))
            troughs(j) = 1;
        end
        %--------------------------------------------------------------------------
    end
else
    for j = 1:N,
        %         index = max(j-rng,1):min(j+rng,N);
        if(j>rng & j<N-rng)
            index = j-rng:j+rng;
        elseif(j>rng)
            index = N-2*rng:N;
        else
            index = 1:2*rng;
        end
        
        if(min(x(index))==x(j))
            peaks(j) = 1;
        end
        
        %-----------------------ѡ�񴰿�����С��Ϊ���ȵ�-----------------------------
        if(max(x(index))==x(j))
            troughs(j) = 1;
        end
        %--------------------------------------------------------------------------
    end
end

%�޳���������ϳ��ȵķ�ֵ�벨��
peaks=find(peaks);
d = diff(peaks);
peaks(d<rng)=[];



% M=peaks;
% N=find(troughs);
% lm=length(M);
% ln=length(N);
%
%  for j=1:lm-1
%       a=find((N<M(j+1))&(N>M(j)));
%       if(isempty(a)==1)
%       [a1 a2]= min(x(M(j):M(j+1)));
%       troughs(M(j)+a2-1)=1;
%       end
%
%  end
troughs=find(troughs);
% for i=2:length(peaks)
%     ind=find(troughs<peaks(i));
%     if isempty(ind) || troughs(ind(end))<peaks(i-1)
%         minV=find(x(1:peaks(i))==min(x(peaks(i-1):peaks(i))));
%         if isempty(ind)
%            troughs=[minV(end),troughs];
%         else
%            troughs=[troughs(1:ind(end)),minV(end),troughs((ind(end)+1):end)];
%         end
%     end
% end

% d = diff(troughs);
% troughs(d<rng)=[];
%---------------------ѡ���벨����������������ʵ�----------------------




for i=2:(length(peaks))
    %ѡ�񲨷�ǰ��һ���źŽ��з�����ѡȡ����Ϊ����ǰrng�ĳ����ڵ���С���λ�ã�������
    ind=find(x(1:peaks(i))==min(x((peaks(i)-rng):peaks(i))));
    
    period_s=x(ind(end):peaks(i));
    
    
    %�Գ���Ϊ��׼���ڷ�ֵ�Ͻ��й��򻯣�ʹ��ֵ���ڳ��ȡ���һ���������ƺ�����
    period_s=Normalized_amp(period_s);
%     if i==58
%         tempS{i-1}= period_s;
%     end
%     if isnan(period_s)
%         ind=find(troughs<peaks(i));
%         lows(i-1)=troughs(ind(end));
%     else
    %��������
    y1=diff(period_s);
    y2=[diff(y1);0];
    curvatureAll=abs(y2./(1+y1.^2).^1.5);
    
    %ѡ��������ֵ���¼λ��
    curvatureAll=meanL3(curvatureAll);
    extrMaxIndex = find(diff(sign(diff(curvatureAll)))==-2)+1;
    temp=zeros(length(curvatureAll),1);
    temp(extrMaxIndex)=curvatureAll(extrMaxIndex);
    curvatureAll=temp;
    
    %ѡ���źż�Сֵ���¼λ��
    extrMinIndex = [1;diff(sign(diff(period_s)))==2];
    
    %ɾ��̫������������ʼ���ֵ���źż�Сֵ
    curvatureAll(find(period_s>(0.5*length(period_s))))=0;
    extrMinIndex(find(period_s>(0.5*length(period_s))))=0;
    
    %�����źŰ�͹�ԣ��𲫵�λ�ڰ���λ�ã�ɾ������λ�õ����ʼ���ֵ��
    ccd=concave_convex_detection(period_s);
    curvatureAll(find(ccd==1))=0;
    curvatureAll(find(ccd==0))=0;
    %ѡ��3����������ֵ��Ϊ��ѡ�и��
    [~,indx]=sort(curvatureAll,'descend');
    indx=indx(1:3);
    indx(find(curvatureAll(indx)==0))=[];
    
    
    indx_m=find(extrMinIndex);
    %�����������һ����������������ֵ
    if  ~isempty(indx)
        %����ѡ����������ʼ���ֵ����źż�Сֵ���������������㣬�Ƚ���һ�����������塣
        %���źż�Сֵ����������壬��ֱ��������źż�Сֵ����Ϊ�и��
        if indx_m(end)>indx(indx==max(indx))-20
            lows(i-1)=indx_m(end);
        else
            %�Ƚ���һ�����ʼ���ֵ���벨������߸�����һ��ֱ��
            L1=Distance(period_s(indx(end):end));
            lows(i-1)=indx(end);
            for k=1:(length(indx)-1)
                L2=Distance(period_s(indx(end-k):end));
                if L2<L1
                    L1=L2;
                    lows(i-1)=indx(end-k);
                end
            end
        end
    else%��������������ֵΪ�գ���ѡ���������ļ�Сֵ����Ϊ�и��
        lows(i-1)=indx_m(end);
    end
    
    %     if ~isempty(indx_m)
    %         s=abs(indx_m-lows(i-1));
    %         t=s(find(s<40));
    %        if ~isempty(t)
    %             s=find(s==min(t));
    %            lows(i-1)=indx_m(s(end));
    %         end
    %     end
    
    lows(i-1)=ind(end)+lows(i-1);
    end

% end
%s=curvatureAll(lows);
lows(find(lows==0))=[];
peaks=peaks;

end

