function [peaks,lows] = PeakDetection_Distance_Normalized( x,ff )
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
%x=Normalized_amp(x);
N = length(x);
peaks = zeros(N,1);
%lows=zeros(N,1);

th = .5;
rng = floor(th/ff);

if(abs(max(x))>abs(min(x)))%max肯定大于min，x中没有负值
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
        %-----------------------选择窗口内最小点为波谷点-----------------------------
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
        
        %-----------------------选择窗口内最小点为波谷点-----------------------------
        if(max(x(index))==x(j))
            troughs(j) = 1;
        end
        %--------------------------------------------------------------------------
    end
end

%剔除间隔不符合长度的峰值与波谷
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
%---------------------选择离波峰距离最近的最大曲率点----------------------




for i=2:(length(peaks))
    %选择波峰前的一段信号进行分析，选取方法为波峰前rng的长度内的最小点的位置，到波峰
    ind=find(x(1:peaks(i))==min(x((peaks(i)-rng):peaks(i))));
    
    period_s=x(ind(end):peaks(i));
    
    
    %以长度为基准，在幅值上进行规则化，使幅值等于长度。这一步的作用似乎不大
    period_s=Normalized_amp(period_s);
%     if i==58
%         tempS{i-1}= period_s;
%     end
%     if isnan(period_s)
%         ind=find(troughs<peaks(i));
%         lows(i-1)=troughs(ind(end));
%     else
    %计算曲率
    y1=diff(period_s);
    y2=[diff(y1);0];
    curvatureAll=abs(y2./(1+y1.^2).^1.5);
    
    %选择曲极大值点记录位置
    curvatureAll=meanL3(curvatureAll);
    extrMaxIndex = find(diff(sign(diff(curvatureAll)))==-2)+1;
    temp=zeros(length(curvatureAll),1);
    temp(extrMaxIndex)=curvatureAll(extrMaxIndex);
    curvatureAll=temp;
    
    %选择信号极小值点记录位置
    extrMinIndex = [1;diff(sign(diff(period_s)))==2];
    
    %删除太靠近波峰的曲率极大值和信号极小值
    curvatureAll(find(period_s>(0.5*length(period_s))))=0;
    extrMinIndex(find(period_s>(0.5*length(period_s))))=0;
    
    %计算信号凹凸性，起搏点位于凹的位置，删除其他位置的曲率极大值点
    ccd=concave_convex_detection(period_s);
    curvatureAll(find(ccd==1))=0;
    curvatureAll(find(ccd==0))=0;
    %选择3个最大的曲率值作为候选切割点
    [~,indx]=sort(curvatureAll,'descend');
    indx=indx(1:3);
    indx(find(curvatureAll(indx)==0))=[];
    
    
    indx_m=find(extrMinIndex);
    %如果存在至少一个符合条件的曲率值
    if  ~isempty(indx)
        %对于选择出来的曲率极大值点和信号极小值点最靠近波峰的两个点，比较哪一个更靠近波峰。
        %若信号极小值点更靠近波峰，则直接以这个信号极小值点作为切割点
        if indx_m(end)>indx(indx==max(indx))-20
            lows(i-1)=indx_m(end);
        else
            %比较哪一个曲率极大值点与波峰的连线更近与一条直线
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
    else%符合条件的曲率值为空，则选择最靠近波峰的极小值点作为切割点
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

