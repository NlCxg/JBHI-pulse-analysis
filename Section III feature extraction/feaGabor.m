function fea = feaGabor( X,dict ,L)


if ~isempty(X)
    X=X/(sum(X.*X));
    fea=OMP(dict,X',L);
else
    fea=zeros(1,64*(L+1));
end

end

