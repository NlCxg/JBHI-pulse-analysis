function [ output_cycles ] = outlier_check(input_cycles,len,L,A)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if(length(input_cycles)>0)
    
%     subplot(311);
%       hold on;
%       for i=1:length(input_cycles);
%       plot(input_cycles{1,i})
%       end
%       hold off;
      
    limitL=0.15*len;
    aL=cellfun(@length,input_cycles);
    input_cycles(abs(aL-len)>limitL)=[];
    aL=cellfun(@length,input_cycles);
    
     
    
    
    mid=mean(aL);
    m_std=std(aL);
    ind=find( aL<mid+3*m_std  &  aL>mid-3*m_std);
    
%      subplot(312);
%       hold on;
%       for i=1:length(input_cycles);
%       plot(input_cycles{1,i})
%       end
%       hold off;
      
    if length(ind)>5
    input_cycles=input_cycles(ind);
    
    
    L=mat2cell(ones(1,length(input_cycles))'*L,ones(1,length(input_cycles)))';
    A=mat2cell(ones(1,length(input_cycles))'*A,ones(1,length(input_cycles)))';
    output_cycles=cellfun(@pulse_normal,input_cycles',L',A','UniformOutput', false);
    output_cycles=cell2mat(output_cycles)';
    
%       subplot(313);
%       hold on;
%       for i=1:length(input_cycles);
%       plot(output_cycles(:,i));
%       end
%       hold off;
      
    cor=corrcoef(output_cycles);
    cor(cor>0.92)=1;
    cor(cor<=0.92)=0;
    cor=sum(cor,2);
    cor=find(cor>(length(cor)*0.5));
    
  
    
    
    if ~isempty(cor)
        output_cycles=output_cycles(:,cor)';
        
%       hold on;
%       for i=1:length(output_cycles);
%          plot(output_cycles(i,:));
%       end
      
    else
        output_cycles=[];
    end
    
    else
         output_cycles=[];
    end
    
    
    
else
    output_cycles=[];
end
end

