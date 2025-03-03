function [normalized_amp] = Normalized_amp( x )
          

x1=min(x);             %·ùÖµ
x2=max(x);
normalized_amp=(x-x1)*length(x)/(x2-x1);
end