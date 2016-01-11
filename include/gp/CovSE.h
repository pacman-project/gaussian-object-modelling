/** @file CovCovSE.h
 * 
 * 
 * @author	Claudio Zito
 *
 * @copyright  Copyright (C) 2015 Claudio, University of Birmingham, UK
 *
 * @license  This file copy is licensed to you under the terms described in
 *           the License.txt file included in this distribution.
 *
 * Refer to Gaussian process library for Machine Learning.
 *
 */
#ifndef __GP__COVSE_H__
#define __GP__COVSE_H__

//------------------------------------------------------------------------------

#include "gp/Covs.h"
#include <cmath>

//------------------------------------------------------------------------------

namespace gp {

//------------------------------------------------------------------------------

class CovSE : public BaseCovFunc {
public:
	typedef boost::shared_ptr<CovSE> Ptr;

	/** Descriptor file */
	class Desc : public BaseCovFunc::Desc {
	public:
		typedef boost::shared_ptr<CovSE::Desc> Ptr;
		double sigma;
        double length;
		
		/** Default C'tor */
		Desc() {
			setToDefault();
		}
		
		/** Set values to default */
		void setToDefault() {
			BaseCovFunc::Desc::setToDefault();
			inputDim = 0;
			paramDim = 2;
			sigma = 1.0;
			length = 1.0;
		}
		
		/** Creates the object from the description. */
		CREATE_FROM_OBJECT_DESC_0(CovSE, BaseCovFunc::Ptr)
		
		/** Assert valid descriptor files */
		bool isValid(){ 
			if (!std::isfinite(sigma) || !std::isfinite(length))
				return false;
			return true;
		}
	};
	
	/** Get name of the covariance functions */
	virtual std::string getName() const {
		return "CovSE";
	}
        
	/** laplacian kernel = sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng)) */
	inline double get(const Vec3& x1, const Vec3& x2) const {
		const double z = ((x1 - x2) / ell).magnitude();//sqrt(DD);
		return sf2*std::exp(-0.5*z);
	}
	/** Compute the kernel */
	virtual double get(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const {
		double z = ((x1 - x2) / ell).squaredNorm();
		return sf2*exp(-0.5*z);
	}
	/** laplacian derivate = -1*invLenght*[sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))] */
	inline double getDiff(const Vec3& x1, const Vec3& x2, const double noise = .0) const {
		const double EE = get(x1, x2);// +noise;//sqrt(DD);
		return -1 * ell * EE;
	}

	/** Covariance gradient of two input vectors with respect to the hyperparameters.
	*  @param x1 first input vector
	*  @param x2 second input vector
	*  @param grad covariance gradient */
	virtual void grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, Eigen::VectorXd& g) const {
		double z = ((x1 - x2) / ell).squaredNorm();
		double k = sf2*exp(-0.5*z);
		g << k*z, 2 * k;
		//double z = loghyper(0) != .0 ? -(x1 - x2).lpNorm<2>()*get(x1, x2) / pow(loghyper(0), 2.0) : -(x1 - x2).lpNorm<2>()*get(x1, x2);//((x1 - x2)*sqrt3 / invLength).norm();
		//const double EE = (double)(x1 - x2).lpNorm<2>();//sqrt(DD);
		//const double power = -1 * EE*invLength;
		//double k = 2 * loghyper(1)*exp(power);
		//g << z, k;
	};

	/** Update parameter vector.
	*  @param p new parameter vector */
	virtual void setLogHyper(const Eigen::VectorXd &p) {
		BaseCovFunc::setLogHyper(p);
		ell = exp(loghyper(0));
		sf2 = exp(2 * loghyper(1)); //2 * loghyper(1) * loghyper(1);//
	}

	~CovSE(){}

private:
	/** Hyper-parameters */
	double ell; // second element of loghyper
    double sf2; // first element of loghyper
        
    /** Create from descriptor */
	void create(const Desc& desc) {
		BaseCovFunc::create(desc);
		loghyper.resize(paramDim);
		loghyper(0) = desc.length;
		loghyper(1) = desc.sigma;
		ell = exp(loghyper(0));
		sf2 = exp(2 * loghyper(1)); //2 * loghyper(1) * loghyper(1);//
	}
        
    CovSE() : BaseCovFunc() {}
};

//------------------------------------------------------------------------------

}



//CovSE(const double sigma, const double length) : BaseCovFunc(),
//                sigma_(sigma),
//                length_(length)
//        {
//        	two_sigma_2 = 2*sigma_*sigma_;
//                inv_length_ = 1.0 / (length_);
//                loghyper_changed = true;
//        }

//        CovSE() : BaseCovFunc(),
//                sigma_(1.0),
//                length_(1.0)
//        {
//        	two_sigma_2 = 2*sigma_*sigma_;
//                inv_length_ = 1.0;
//                loghyper_changed = true;
//        }

#endif

