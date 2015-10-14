
function [ Cov_matrix ] = Cov_inv_multiquadric( x_post, leng )
% Computing the covariance matrix using 
% the inverse multiquadric kernel
%
% x_post \in R^{n \times d} is a amtrix containing the 'd' coordinates of
% the 'n' points of which the prediction is desired
% inverse multiquadric kernel =  1/ sqrt(|x_i-x_j|^2+leng)
%           
A = x_post  ;
B = x_post ;
[m,p1] = size(A); [n,p2] = size(B);
AA = sum(A.*A,2);  % column m_by_1
BB = sum(B.*B,2)'; % row 1_by_n
DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
%
Cov_matrix = 1./ sqrt( DD+ones(size(DD)) )   ;
end

