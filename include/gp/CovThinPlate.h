/** @file CovThinPlate.h
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
#ifndef __GP__THINPLATE_H__
#define __GP__THINPLATE_H__

//------------------------------------------------------------------------------

#include "gp/Covs.h"

//------------------------------------------------------------------------------

namespace gp
{

//------------------------------------------------------------------------------

class ThinPlate : public BaseCovFunc 
{
public:
	/** Pointer to the Covariance function */
	typedef boost::shared_ptr<ThinPlate> Ptr;

	/** Descriptor file */
	class Desc : public BaseCovFunc::Desc {
	public:
		/** Pointer to description file */
		typedef boost::shared_ptr<ThinPlate::Desc> Ptr;
		
		/** Hyper-parameters */
        	double length;
		
		/** Default C'tor */
		Desc() {
			setToDefault();
		}
		
		/** Set values to default */
		void setToDefault() {
			length = 1.0;
		}
		
		/** Creates the object from the description. */
		CREATE_FROM_OBJECT_DESC_0(ThinPlate, BaseCovFunc::Ptr)
		
		/** Assert valid descriptor files */
		bool isValid(){ 
			if (!std::isfinite(length))
				return false;
			return true;
		}
	}; 
	
	/** Get name of the covariance functions */
	virtual std::string getName() const {
		return "Laplace";
	}
	
        //thin plate kernel = 2.*EE.^3 - 3.*(leng).* EE.^2 + (leng*ones(size(EE))).^3
        inline double get(const Vec3& x1, const Vec3& x2) const {
        	const double EE = x1.distance(x2);
        	return 2*std::pow(EE, 3.0) - threeLength*pow(EE, 2.0) + length3;
        }

        ~ThinPlate() {};

private:
        /** Hyper-parameters */
        double threeLength;
        double length3;
        
        /** Create from descriptor */
	void create(const Desc& desc) {
		BaseCovFunc::create(desc);
		threeLength = 3*desc.length;
		length3 = std::pow(desc.length, 3.0);
	}
};

//------------------------------------------------------------------------------

}

#endif
