

function [ Cov_matrix ] = Cov_matern_32( x_post, leng , sigma_f)
% Computing the covariance matrix using 
% the inverse multiquadric kernel
%
% x_post \in R^{n \times d} is a amtrix containing the 'd' coordinates of
% the 'n' points of which the prediction is desired
% matern_32 kernel =  sigma_f^2 *(1+sqrt(3)* |x_i-x_j|/leng)*...
%                   exp(-sqrt(3)* |x_i-x_j|/leng    )
%           


A = x_post  ;
B = x_post ;
[m,p1] = size(A); [n,p2] = size(B);
AA = sum(A.*A,2);  % column m_by_1
BB = sum(B.*B,2)'; % row 1_by_n
DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
EE = sqrt(DD) ;
%
Cov_matrix =  sigma_f^2 *(ones(size(DD))+sqrt(3).* DD./leng)*...
                         exp(-sqrt(3).* DD/leng    )  ;
end

