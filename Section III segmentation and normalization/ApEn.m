function ent_val = ApEn(data, m,r)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%samplEntropy( m,r,data)计算时间序列data的近似熵
%%%输入：data为输入数据序列
%%%      m为初始分段
%%%      r为阈值
%%%输出：样本熵值
%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_data=data;
N = length(p_data);
m=2;
%r=0.1;
Nkx1 = 0;
Nkx2 = 0;
clear dx1 dx2 x1 x1temp x2 x2temp
% 分段计算距离，x1为长度为m的序列，x2为长度为m+1的序列
for k = N - m:-1:1
    x1(k, :) = p_data(k:k + m - 1);
    x2(k, :) = p_data(k:k + m);
end

for k = N - m:-1:1
    % x1序列计算
    % 统计距离，由于每行都要与其他行做减法，因此可以先将该行复制为N-m的矩阵，然后
    % 与原始x1矩阵做减法，可以避免两重循环，增加效率
    x1temprow = x1(k, :);
    x1temp    = ones(N - m, 1)*x1temprow;
    % 可以使用repmat函数完成上面的语句，即
    % x1temp = repmat(x1temprow, N - m, 1);
    % 但是效率不如上面的矩阵乘法
    % 计算距离，每一行元素相减的最大值为距离
    dx1(k, :) = max(abs(x1temp - x1), [], 2)';
    % 模板匹配数
    if log( (sum(dx1(k, :) < r) - 1)/(N - m +1))~=-Inf
        Nkx1 = Nkx1 +log( (sum(dx1(k, :) < r) - 1)/(N - m +1));
    end
    
    % x2序列计算，和x1同样方法
    x2temprow = x2(k, :);
    x2temp    = ones(N - m, 1)*x2temprow;
    dx2(k, :) = max(abs(x2temp - x2), [], 2)';
    if log( (sum(dx2(k, :) < r) - 1)/(N - m ))~=-Inf
        Nkx2 = Nkx2+log( (sum(dx2(k, :) < r) - 1)/(N - m));
    end
end
% 平均值
Bmx1 = Nkx1/(N - m+1);
Bmx2 = Nkx2/(N - m);
% 样本熵
ent_val = Bmx1-Bmx2;
end

