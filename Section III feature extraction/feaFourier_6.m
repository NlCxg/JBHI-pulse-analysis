function fea = feaFourier_6(cycle2)
%CREATEFIT(CYCLE2)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      Y Output: cycle2
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 23-Mar-2017 14:27:03 自动生成


%% Fit: 'untitled fit 1'.
if ~isempty(cycle2)
[xData, yData] = prepareCurveData( [], cycle2 );

% Set up fittype and options.
ft = fittype( 'fourier6' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0997331001139617];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );
fea=[fitresult.a1 fitresult.b1 ...
    fitresult.a2 fitresult.b2 ...
     fitresult.a3 fitresult.b3 ...
     fitresult.a4 fitresult.b4 ...
      fitresult.a5 fitresult.b5 ...
       fitresult.a6 fitresult.b6 ...
     fitresult.w];
 
% Plot fit with data.
% figure( 'Name', 'untitled fit 1' );
% h = plot( fitresult, xData, yData );
% legend( h, 'cycle2', 'untitled fit 1', 'Location', 'NorthEast' );
% % Label axes
% ylabel cycle2
% grid on

else
    fea=zeros(1,13);
end
