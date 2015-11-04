/* MATLAB
 * function [ Cov_matrix ] = Cov_laplace( x_post, leng, sigma_f )
 * % Computing the covariance matrix using 
 * % the Gaussian kernel
 * %
 * % x_post \in R^{n \times d} is a matrix containing the 'd' coordinates of
 * % the 'n' points of which the prediction is desired
 * % laplacian kernel = sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))
 * %
 * A = x_post  ;
 * B = x_post ;
 * [m,p1] = size(A); [n,p2] = size(B);
 * AA = sum(A.*A,2);  % column m_by_1
 * BB = sum(B.*B,2)'; % row 1_by_n
 * DD = AA(:,ones(1,n)) + BB(ones(1,m),:) - 2*A*B';
 * %
 * Cov_matrix = abs(sigma_f)*exp(sqrt(DD)./(-abs(leng))) ;
 * end
 *
 *
 */
 
 #ifndef GP_REGRESSION___LAPLACE_H
#define GP_REGRESSION___LAPLACE_H

#include <cmath>

namespace gp_regression
{

class Laplace
{
public:
        const double sigma_;
        const double length_;

        inline double compute(double &value)
        {
                double power = -1*value*inv_length_;
                double out = 2*sigma_*std::exp(power);
                return out;
        }

        Laplace(double sigma, double length) :
                sigma_(sigma),
                length_(length)
        {
                inv_length_ = 1.0 / (length_);
        }

        Laplace() :
                sigma_(1.0),
                length_(1.0)
        {
                inv_length_ = 1.0;
        }

private:
        double inv_length_;

};

}

#endif
