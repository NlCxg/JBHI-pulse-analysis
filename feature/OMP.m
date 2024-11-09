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

    for j=1:1:L
        proj=D'*residual;
        [maxVal,pos]=max(abs(proj));
        pos=pos(1);
        indx(j)=pos;
        a=pinv(D(:,indx(1:j)))*x;
        residual=x-D(:,indx(1:j))*a;
        A(:,j)=D(:,indx(j))*a(j);

    end

  A=A(:)';

   
end
end