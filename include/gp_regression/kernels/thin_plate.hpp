#ifndef GP_REGRESSION___THINPLATE_H
#define GP_REGRESSION___THINPLATE_H

#include <cmath>

namespace gp_regression
{

class ThinPlate
{
public:
        const double R_;

        inline double compute(double &value)
        {
                double pos = std::abs(value);
                return 2*pos*pos*pos - 3*R_*value*value + R3_;
        }

        inline double computediff(double &value)
        {
                double pos = std::abs(value);

                double out = pos*pos*(pos/value) - R_*value;
                if( std::isnan(out) )
                        out = R_*value;
                // std::cout << "out( " << value << " ) = " << 6*out << std::endl;
                return 6*out;
        }

        inline double computediffdiff(double &value)
        {
                return 12*value - 6*R_;
        }

        ThinPlate(double R) :
                R_(R)
        {
                R3_ = R_*R_*R_;
        }

        ThinPlate() :
                R_(1.0)
        {
                R3_ = 1.0;
        }

private:
        double R3_;
};

}

#endif
