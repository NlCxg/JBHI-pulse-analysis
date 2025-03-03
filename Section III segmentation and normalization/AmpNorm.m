function a=AmpNorm(b,A)

if length(A)==1

    [a,~]=mapminmax(b',0,A);
end
if length(A)==2
    [a,~]=mapminmax(b',A(1), A(2));
end
a=a';
end
