/** @file SampleSet.h
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
#ifndef __SAMPLESET_H__
#define __SAMPLESET_H__

//------------------------------------------------------------------------------

#include <Eigen/Dense>
#include <gp/Vec3.h>
#include <boost/shared_ptr.hpp>

//------------------------------------------------------------------------------

namespace gp {

//------------------------------------------------------------------------------

class SampleSet
{
public:
	typedef boost::shared_ptr<SampleSet> Ptr;
	/** Default C'tor. Does nothing.
	*/
	SampleSet();
	/** C'tor.
	*   @param x input 
	*/
	SampleSet(const Vec3Seq& inputs, const RealSeq& targets, const Vec3Seq& normals);
	
	/** Destructor. Does nothing */
	virtual ~SampleSet();
	
	/** Add input-output patterns to sample set.
	* @param x input array
	* @param y target values */
	void add(const Vec3Seq& newInputs, const RealSeq& newTargets, const Vec3Seq& newNormals);
	
	/** Get input vector at index k. */
	//const Eigen::VectorXd & x(size_t k);
	inline const Vec3& x(size_t k) const { 
		return X[k];
	}
	
	/** Get target value at index k. */
	inline double y(size_t k) const {
		return Y[k];
	}
	/** Set target value at index i. */
	bool set_y(size_t i, double y);
	
	/** Get reference to vector of target values. */
	const RealSeq y() const {
		return Y;
	}
	
	/** Get number of samples. */
	inline const size_t rows() const { return n; };
	/** Get dim of samples. */
	inline const size_t cols() const { return 3; };
	
	/** Clear sample set. */
	void clear();
	
	/** Check if sample set is empty. */
	inline bool empty () const { return n==0; };

private:
	/** Container holding input vectors. */
	Vec3Seq X;
	/** Container holding target values. */
	RealSeq Y;
	/** Number of samples. */
	size_t n;
};

//------------------------------------------------------------------------------

}
#endif /* __SAMPLESET_H__ */
