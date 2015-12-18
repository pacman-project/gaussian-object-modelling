/** @file Covs.h
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
#ifndef ___GP_COV_FUNCTIONS_H
#define ___GP_COV_FUNCTIONS_H

//------------------------------------------------------------------------------

#include <cmath>
#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

//------------------------------------------------------------------------------

namespace gp
{

//------------------------------------------------------------------------------

static Eigen::VectorXd convertToEigen(const Vec& v) {
	return Eigen::Map<Eigen::VectorXd>((double *)v.data(), v.size());
}
static Eigen::Vector3d convertToEigen(const Vec3& v) {
	return Eigen::Map<Eigen::VectorXd>((double *)v.get(), 3);
}

//------------------------------------------------------------------------------

/** Object creating function from the description. */
#define CREATE_FROM_OBJECT_DESC_0(OBJECT, POINTER) virtual POINTER create() const {\
	OBJECT *pObject = new OBJECT();\
	POINTER pointer(pObject);\
	pObject->create(*this);\
	return pointer;\
}

/** Object creating function from the description. */
#define CREATE_FROM_OBJECT_DESC_1(OBJECT, POINTER, PARAMETER) virtual POINTER create(PARAMETER parameter) const {\
	OBJECT *pObject = new OBJECT(parameter);\
	POINTER pointer(pObject);\
	pObject->create(*this);\
	return pointer;\
}

/** Template bject creating function from the description. */
#define CREATE_FROM_OBJECT_TEMPLATE_DESC_1(OBJECT, TEMPLATE, POINTER, PARAMETER) virtual POINTER create(PARAMETER parameter) const {\
	OBJECT<TEMPLATE> *pObject = new OBJECT<TEMPLATE>(parameter);\
	POINTER pointer(pObject);\
	pObject->create(*this);\
	return pointer;\
}

//------------------------------------------------------------------------------

class BaseCovFunc
{
public:
	typedef boost::shared_ptr<BaseCovFunc> Ptr;
	
	/** Descriptor file */
	class Desc {
	public:
		typedef boost::shared_ptr<BaseCovFunc::Desc> Ptr;
		/** Default C'tor */
		Desc() {
			setToDefault();
		}
		
		/** Set values to default */
		void setToDefault() {}
		
		/** Creates the object from the description. */
		CREATE_FROM_OBJECT_DESC_0(BaseCovFunc, BaseCovFunc::Ptr)
		
		/** Assert valid descriptor files */
		bool isValid(){ return true; }
		
	};
	
	/** Get name of the covariance functions */
	virtual std::string getName() const {
		return "BaseCovFunc";
	}
	  
        /** Compute the kernel */
        virtual inline double get(const Vec3& x1, const Vec3& x2) const { return x1.distance(x2); }
        /** Compute the derivate */
	virtual inline double getDiff(const Vec3& x1, const Vec3& x2) const { return x1.distanceSqr(x2)*2; }

        /** Access to loghyper_change */
        inline bool getLogHyper() const { return loghyper_changed; }
        inline void setLogHyper(const bool b) { loghyper_changed = b; }
        
        virtual ~BaseCovFunc() {};

protected:
	/** Determine when to recompute the kernel */
	bool loghyper_changed;
	
	/** Create from descriptor */
        virtual void create(const Desc& desc) {
        	loghyper_changed = true;
        }
        
	/** Default C'tor */
        BaseCovFunc() {}
};

//------------------------------------------------------------------------------

}

#endif
