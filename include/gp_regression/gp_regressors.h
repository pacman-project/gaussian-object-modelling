#ifndef GP_REGRESSION___GP_REGRESSORS_H
#define GP_REGRESSION___GP_REGRESSORS_H

#include <gp_regression/gp_regressor.hpp>

// Convenience typedefs. Note that these will use the default constructors!

namespace gp_regression
{

class GaussianRegressor : public GPRegressor<gp_regression::Gaussian> {};
class LaplaceRegressor : public GPRegressor<gp_regression::Laplace> {};
class ThinPlateRegressor : public GPRegressor<gp_regression::ThinPlate>
{
    public:
    typedef std::shared_ptr<ThinPlateRegressor> Ptr;
    typedef std::shared_ptr<const ThinPlateRegressor> ConstPtr;
};

}

#endif
