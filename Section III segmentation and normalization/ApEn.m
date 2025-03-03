function ent_val = ApEn(data, m,r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%samplEntropy( m,r,data)����ʱ������data�Ľ�����
%%%���룺dataΪ������������
%%%      mΪ��ʼ�ֶ�
%%%      rΪ��ֵ
%%%�����������ֵ
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_data=data;
N = length(p_data);
m=2;
%r=0.1;
Nkx1 = 0;
Nkx2 = 0;
clear dx1 dx2 x1 x1temp x2 x2temp
% �ֶμ�����룬x1Ϊ����Ϊm�����У�x2Ϊ����Ϊm+1������
for k = N - m:-1:1
    x1(k, :) = p_data(k:k + m - 1);
    x2(k, :) = p_data(k:k + m);
end

for k = N - m:-1:1
    % x1���м���
    % ͳ�ƾ��룬����ÿ�ж�Ҫ������������������˿����Ƚ����и���ΪN-m�ľ���Ȼ��
    % ��ԭʼx1���������������Ա�������ѭ��������Ч��
    x1temprow = x1(k, :);
    x1temp    = ones(N - m, 1)*x1temprow;
    % ����ʹ��repmat��������������䣬��
    % x1temp = repmat(x1temprow, N - m, 1);
    % ����Ч�ʲ�������ľ���˷�
    % ������룬ÿһ��Ԫ����������ֵΪ����
    dx1(k, :) = max(abs(x1temp - x1), [], 2)';
    % ģ��ƥ����
    if log( (sum(dx1(k, :) < r) - 1)/(N - m +1))~=-Inf
        Nkx1 = Nkx1 +log( (sum(dx1(k, :) < r) - 1)/(N - m +1));
    end
    
    % x2���м��㣬��x1ͬ������
    x2temprow = x2(k, :);
    x2temp    = ones(N - m, 1)*x2temprow;
    dx2(k, :) = max(abs(x2temp - x2), [], 2)';
    if log( (sum(dx2(k, :) < r) - 1)/(N - m ))~=-Inf
        Nkx2 = Nkx2+log( (sum(dx2(k, :) < r) - 1)/(N - m));
    end
end
% ƽ��ֵ
Bmx1 = Nkx1/(N - m+1);
Bmx2 = Nkx2/(N - m);
% ������
ent_val = Bmx1-Bmx2;
end

