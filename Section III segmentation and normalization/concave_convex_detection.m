function [ccd]=concave_convex_detection(p)


n=length(p);
p=[[1:n]',p];
ccd=zeros(n,1);
for i=1:n    
   if i==1                     %�����һ����
        v1=p(n,:)-p(1,:);       %��ǰ�㵽ǰһ������
        v2=p(2,:)-p(1,:);       %��ǰ�㵽��һ������
    elseif i==n                 %���һ����
        v1=p(n-1,:)-p(n,:);    
        v2=p(1,:)-p(n,:);        
    else                        %������
        v1=p(i-1,:)-p(i,:);     
        v2=p(i+1,:)-p(i,:);
    end
    r=det([v1;v2]);                 %��˺�����������ķ���
    if r>0
       ccd(i)=1;
    elseif r<0
        ccd(i)=-1;
    end
end

end