function A=OMP(D,X,L)

if nargin==2
L=size(D,2);
end
P=size(X,2);
K=size(D,2);



for k=1:1:P
    a=[];
    x=X(:,k);
    residual=x;
    indx=zeros(L,1);
    
%     subplot(221);
%      plot(X);
    
    for j=1:1:L
        proj=D'*residual;
        [maxVal,pos]=max(abs(proj));
        pos=pos(1);
        indx(j)=pos;
        a=pinv(D(:,indx(1:j)))*x;
        residual=x-D(:,indx(1:j))*a;
        A(:,j)=D(:,indx(j))*a(j);
%         if sum(residual.^2)<1e-6
%             break;
%         end
       
%         subplot(222);
%         plot(D(:,pos));
%         
%         subplot(223);
%         plot(X-residual);
%         
%         subplot(224);
%         plot(residual);
        
    end
   %  temp=zeros(K,1);
   %  temp(indx(1:j))=a;
   %  A(:,k)=sparse(temp);
  % A=[indx',a'];
  A=A(:)';
  %%%%%%%%%%%
%     figure;
%   plot(X);
%   
%   figure;
%  subplot(231);
%  plot(A(1:64));
%   subplot(232);
%  plot(A(65:64*2));
%   subplot(233);
%  plot(A(64*2+1:64*3));
%   subplot(234);
%  plot(A(64*3+1:64*4));
%   subplot(235);
%  plot(A(64*4+1:64*5));
%   subplot(236);
%  plot(A(64*5+1:64*6));
%  
%  figure;
%  plot(A(1:64)+A(65:64*2)+A(64*2+1:64*3)+A(64*3+1:64*4)+A(64*4+1:64*5));
%%%%%%%%%%%%%%%
   
end
end
