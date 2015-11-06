/** @file CovLaplace.h
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
	typedef boost::shared_ptr<Laplace> Ptr;

	/** Descriptor file */
	class Desc : public BaseCovFunc::Desc {
	public:
		typedef boost::shared_ptr<Laplace::Desc> Ptr;
		double sigma;
        	double length;
		
		/** Default C'tor */
		Desc() {
			setToDefault();
		}
		
		/** Set values to default */
		void setToDefault() {
			sigma = 1.0;
			length = 1.0;
		}
		
		/** Creates the object from the description. */
		CREATE_FROM_OBJECT_DESC_0(Laplace, BaseCovFunc::Ptr)
		
		/** Assert valid descriptor files */
		bool isValid(){ 
			if (!std::isfinite(sigma) || !std::isfinite(length))
				return false;
			return true;
		}
	};
	
	/** Get name of the covariance functions */
	virtual std::string getName() const {
		return "Laplace";
	}
        
        //laplacian kernel = sigma_f^2*exp(sqrt((p_1-p_2)'*(p_1-p_2))/(-leng))
        inline double get(const Vec3& x1, const Vec3& x2) const {
        	const double EE = x1.distance(x2);//sqrt(DD);
        	const double power = -1*EE*invLength;
        	return twoSigma2*std::exp(power);
        }
	
	~Laplace(){}
private:
	/** Hyper-parameters */
	double twoSigma2;
        double invLength;
        
        /** Create from descriptor */
	void create(const Desc& desc) {
		BaseCovFunc::create(desc);
		twoSigma2 = 2*std::pow(desc.sigma, 2.0);
                invLength = 1.0 / desc.length;
	}
        
        Laplace() : BaseCovFunc() {}
};

//------------------------------------------------------------------------------

}



//Laplace(const double sigma, const double length) : BaseCovFunc(),
//                sigma_(sigma),
//                length_(length)
//        {
//        	two_sigma_2 = 2*sigma_*sigma_;
//                inv_length_ = 1.0 / (length_);
//                loghyper_changed = true;
//        }

//        Laplace() : BaseCovFunc(),
//                sigma_(1.0),
//                length_(1.0)
//        {
//        	two_sigma_2 = 2*sigma_*sigma_;
//                inv_length_ = 1.0;
//                loghyper_changed = true;
//        }

#endif
