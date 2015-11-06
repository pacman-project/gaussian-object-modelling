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
#ifndef __GP__LAPLACE_H__
#define __GP__LAPLACE_H__

//------------------------------------------------------------------------------

#include "gp/Covs.h"

//------------------------------------------------------------------------------

namespace gp
{

//------------------------------------------------------------------------------

class Laplace : public BaseCovFunc {
public:
        const double sigma_;
        const double length_;
//        inline double compute(double &value)
//        {
//                double power = -1*value*inv_length_;
//                double out = 2*sigma_*std::exp(power);
//                return out;
//        }
        
        //laplacian kernel = sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))
        double get(const Vec3& x1, const Vec3& x2) const {
//        	const double sum_x1_2 = x1.magnitudeSqr();
//        	const double sum_x2_2 = x2.magnitudeSqr();
//        	const double DD = sum_x1_2 + sum_x2_2 - 2*x1.dot(x2);
        	const double EE = x1.distance(x2);//sqrt(DD);
        	const double power = -1*EE*inv_length_;
        	return 2*sigma_2*std::exp(power);
//        	Vec3 d = x1 - x2;
//        	double value = std::sqrt(d.dot(d));
//        	double power = -1*value*inv_length_;
//                double out = 2*sigma_*std::exp(power);
        }

        Laplace(const double sigma, const double length) : BaseCovFunc(),
                sigma_(sigma),
                length_(length)
        {
        	sigma_2 = sigma_*sigma_;
                inv_length_ = 1.0 / (length_);
                loghyper_changed = true;
        }

        Laplace() : BaseCovFunc(),
                sigma_(1.0),
                length_(1.0)
        {
        	sigma_2 = sigma_*sigma_;
                inv_length_ = 1.0;
                loghyper_changed = true;
        }

private:
	double sigma_2;
        double inv_length_;
};

//------------------------------------------------------------------------------

}

#endif
