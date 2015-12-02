#ifndef GP_REGRESSION___GAUSSIAN_H
#define GP_REGRESSION___GAUSSIAN_H

#include <cmath>

namespace gp_regression
{

class Gaussian
{
public:
        const double sigma_;
        const double length_;

        inline double compute(double &value)
        {
                double power = -1*value*inv_length2_;
                double out = sigma2_*std::exp(power);
                return out;
        }

        inline double computediff(double &value)
        {
                double e = compute(value);
                double out = -1*inv_length2_*e/value;
                return out;
        }

        inline double computediff2(double &value)
        {
                // double e = compute(value);
                // double out = inv_length2_*inv_length2_*e;
                return 0.0;
        }

        Gaussian(double sigma, double length) :
                sigma_(sigma),
                length_(length)
        {
                sigma2_ = sigma_ * sigma_;
                inv_length2_ = 1.0 / (length_ * length_);
        }

        Gaussian() :
                sigma_(1.0),
                length_(1.0)
        {
                sigma2_ = 1.0;
                inv_length2_ = 1.0;
        }

private:
        double sigma2_;
        double inv_length2_;

};

}

#endif
