#ifndef GP_REGRESSION___GP_REGRESSOR_H
#define GP_REGRESSION___GP_REGRESSOR_H

#include <cassert>
#include <string>
#include <vector>

#include <gp_regression/cov_functions.h>
#include <gp_regression/gp_regression_exception.h>

namespace gp_regression
{

/*
 * \brief Container for input data representing the raw 3D points
 */
struct Data
{
	std::vector<double*> coord_x;
	std::vector<double*> coord_y;
	std::vector<double*> coord_z;
};

/*
 * \brief Container for model parameters that represent an object
 */
struct Model
{
	std::vector<double*> mean_x;
	std::vector<double*> mean_y;
	std::vector<double*> mean_z;
	std::vector<double*> std_dev_x;
	std::vector<double*> std_dev_y;
	std::vector<double*> std_dev_z;
	std::vector<double*> weight;
};


/*
 * \brief Handle for propagating a single GP map from 3D points to paramters, 
 * from-to parameters of 3D points, sample points, etc. 
 *
 */
template <class CovType>
class GPRegressor
{
public:
	virtual ~GPRegressor() {}

	// CovType getCovType()
	// {
		// try
		// {
		// 	// return this->
		// }
		// catch(const std::logic_error& e)
		// {
		// 	//throw TransmissionInterfaceException(e.what());
		// }
	// }

	/**
	* \brief Solves the regression problem, computes the model parameters.
	* \param[in]  data 3D points.
	* \param[out] gp Gaussian Process parameters.
	* \pre All non-empty vectors must contain valid data and their size 
	* should be equal among them. Data vectors not used in this function can
	* remain empty, so they can be used for 3D or 2D. For now, only 3D is
	* assumed.
	*/
	void generateModel(const Data& data, Model& gp)
	{

	}

	/** 
	 * \brief Generates data using the GP
	 * \param[in]  data 3D points.
	 */
	void generateData(const Model& gp, Data& data)
	{

	}

	void updateModel(const Data& data, Model& gp)
	{

	}

	/* 
	 * \return Number of example points in the GP
	 */
	virtual std::size_t numPoints() const = 0;

private:

};

// Useful typedefs

class GaussianRegressor : public GPRegressor<gp_regression::Gaussian> {};
// class LaplaceRegressor : public GPRegressor<Laplace> {};
// class InvMultiQuadRegressor : public GPRegressor<InvMultiQuad> {};

}

#endif