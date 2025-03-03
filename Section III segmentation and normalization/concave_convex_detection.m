function [ccd]=concave_convex_detection(p)


n=length(p);
p=[[1:n]',p];
ccd=zeros(n,1);
for i=1:n    
   if i==1                     %处理第一个点
        v1=p(n,:)-p(1,:);       %当前点到前一点向量
        v2=p(2,:)-p(1,:);       %当前点到后一点向量
    elseif i==n                 %最后一个点
        v1=p(n-1,:)-p(n,:);    
        v2=p(1,:)-p(n,:);        
    else                        %其他点
        v1=p(i-1,:)-p(i,:);     
        v2=p(i+1,:)-p(i,:);
    end
    r=det([v1;v2]);                 %叉乘后第三个向量的方向
    if r>0
       ccd(i)=1;
    elseif r<0
        ccd(i)=-1;
    end
end

end