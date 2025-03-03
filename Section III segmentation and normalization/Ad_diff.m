function [tt,pp,fp,B] =Ad_diff(a)

apre=m_denoise(a);

extrMinIndex = find(diff(sign(diff(apre)))==+2)+1;
extrMaxIndex = find(diff(sign(diff(apre)))==-2)+1;
extrInflectionIndex = find(diff(sign(diff(apre)))==0)+1;
thr=mean(apre);


%判断信号正反向，以信号均值为界，复杂性较高的一面是波谷位置，较小的一面是波峰位置
%两种方法：
%1，信号长度在边界两侧分布不均，长的是波谷
%2，极值点数量在边界两侧分布不均，多的是波谷
%初步试验第一种方法比第二种要好一点。但对于某些信号，两种方法都不是很好。
upperL= length(find(apre(extrMinIndex)>thr))+length(find(apre(extrMaxIndex)>thr));
lowerL= length(find(apre(extrMinIndex)<thr))+length(find(apre(extrMaxIndex)<thr));

upperAll=length(find(apre>thr));
lowerAll=length(find(apre<thr));
 if upperAll>lowerAll
     apre=max(apre)-apre+min(apre);
 end



     fp=periodNum(apre);%求周期长度
     if(fp<180||fp>650)%对于不满足长度的信号直接排除
        tt={};
        pp={};
        B=apre(200:(end-200));
        return;
     else
          fh=1/fp;
     end
     

       
     [peaks, lows]=PeakDetection_Distance_Normalized(apre,fh);%求波峰波谷
     
     %去基线漂移
     xi=1:1:length(apre);
     
     yi=apre(lows);
     lows=double(lows);
     base=interp1(lows,yi,xi,'spline')';
     B=apre-base+mean(apre(lows));
     B(B<0)=0;
     B=B(200:(end-200));
    
    [peaks,lows]=PeakDetection_Distance_Normalized(B,fh);
     %pp为峰到峰的分割，tt为谷到谷的分割
      [pp,tt]=periodseg_modified(peaks,lows,B,fp);
    

end
