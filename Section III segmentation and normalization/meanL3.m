function C=meanL3(A)
C=A;
for i=2:length(A)-1
    C(i)=(A(i-1)+A(i)+A(i+1))/3;
end

    