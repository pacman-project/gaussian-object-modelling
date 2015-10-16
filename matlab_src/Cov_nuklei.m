function [ Cov_matrix ] = Cov_nuklei( x_post, leng  )
% Computing the covariance matrix using 
% the Gaussian kernel
%
% x_post \in R^{n \times d} is a matrix containing the 'd' coordinates of
% the 'n' points of which the prediction is desired
% nuklei kernel = 1 -  (|x_i-x_j|)/(2*leng) if (|x_i-x_j|)/(leng)<= 1
%               = 0 otherwise
A = x_post  ;
B = x_post ;
[m,p1] = size(A); [n,p2] = size(B);
AA = sum(A.*A,2);  % column m_by_1
BB = sum(B.*B,2)'; % row 1_by_n
DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
EE = sqrt(DD) ;
%
Cov_matrix = ones(size(EE))-EE./(abs(leng)) ;
Cov_matrix(Cov_matrix<=0)=0 ;
end

