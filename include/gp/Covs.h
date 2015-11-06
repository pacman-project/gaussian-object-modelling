#ifndef ___GP_COV_FUNCTIONS_H
#define ___GP_COV_FUNCTIONS_H

//------------------------------------------------------------------------------

#include <cmath>

//------------------------------------------------------------------------------

namespace gp
{

//------------------------------------------------------------------------------

class BaseCovFunc
{
public:
        bool loghyper_changed;
        
        //laplacian kernel = sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))
        virtual double get(const Vec3& x1, const Vec3& x2) const = 0;

        BaseCovFunc() {
                loghyper_changed = true;
        }
};

//------------------------------------------------------------------------------

}

#endif
