function [ Cov_matrix ] = Cov_gauss( x_post, leng, sigma_f )
% Computing the covariance matrix using 
% the Gaussian kernel
%
% x_post \in R^{n \times d} is a matrix containing the 'd' coordinates of
% the 'n' points of which the prediction is desired
% gaussian kernel = sigma_f^2*e^(-|x_i-x_j|^2/leng^2)
%
A = x_post  ;
B = x_post ;
[m,p1] = size(A); [n,p2] = size(B);
AA = sum(A.*A,2);  % column m_by_1
BB = sum(B.*B,2)'; % row 1_by_n
DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
%
Cov_matrix = sigma_f^2*exp(DD./(-2*leng^2)) ;
end

