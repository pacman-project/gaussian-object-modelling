#ifndef GP_REGRESSION___THINPLATE_H
#define GP_REGRESSION___THINPLATE_H

#include <cmath>

namespace gp_regression
{

class ThinPlate
{
public:
        inline double compute(double &value)
        {
                return 2*value*value*value - 3*R_*value*value + R3_;
        }

        inline double computediff(double &value)
        {
                return -6*(R_ - value);
        }

        inline double computediffdiff(double &value)
        {
                // not implemented
                return 0;
        }

        ThinPlate(double R) :
                R_(R)
        {
                R3_ = R*R*R;
        }

        ThinPlate() :
                R_(1.0)
        {
                R3_ = 1.0;
        }

private:
        double R3_;
        double R_;
};

}

#endif
