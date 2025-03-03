function [tt,pp,fp,B]=BaselineAd_diff(a)

apre=m_denoise(a);

extrMinIndex = find(diff(sign(diff(apre)))==+2)+1;
extrMaxIndex = find(diff(sign(diff(apre)))==-2)+1;
extrInflectionIndex = find(diff(sign(diff(apre)))==0)+1;
thr=mean(apre);

%�ж��ź����������źž�ֵΪ�磬�����Խϸߵ�һ���ǲ���λ�ã���С��һ���ǲ���λ��
%���ַ�����
%1���źų����ڱ߽�����ֲ������������ǲ���
%2����ֵ�������ڱ߽�����ֲ�����������ǲ���
%���������һ�ַ����ȵڶ���Ҫ��һ�㡣������ĳЩ�źţ����ַ��������Ǻܺá�
upperL= length(find(apre(extrMinIndex)>thr))+length(find(apre(extrMaxIndex)>thr));
lowerL= length(find(apre(extrMinIndex)<thr))+length(find(apre(extrMaxIndex)<thr));

upperAll=length(find(apre>thr));
lowerAll=length(find(apre<thr));
 if upperAll>lowerAll
     apre=max(apre)-apre+min(apre);
 end

     fp=periodNum(apre);%�����ڳ���
     if(fp<180||fp>650)%���ڲ����㳤�ȵ��ź�ֱ���ų�
        tt={};
        pp={};
        B=apre(200:(end-200));
        return;
     else
          fh=1/fp;
     end
     
    
     [peaks, lows]=PeakDetection_Distance_Normalized(apre,fh);%�󲨷岨��
     
     %ȥ����Ư��
     xi=1:1:length(apre);
     
     yi=apre(lows);
     lows=double(lows);
     base=interp1(lows,yi,xi,'spline')';
     B=apre-base+mean(apre(lows));
     B(B<0)=0;
     B=B(200:(end-200));
    
     [peaks,lows]=PeakDetection_Distance_Normalized(B,fh);
     %ppΪ�嵽��ķָttΪ�ȵ��ȵķָ�
      [pp,tt]=periodseg_modified(peaks,lows,B,fp);
  
end
