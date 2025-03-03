function fea = feaGabor( X,dict ,L)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
if ~isempty(X)
    X=X/(sum(X.*X));
    fea=OMP(dict,X',L);
else
    fea=zeros(1,64*(L+1));
end

end

