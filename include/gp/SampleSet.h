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
	SampleSet(const Vec3Seq& inputs, const Vec& targets);
	
	/** Destructor. */
	virtual ~SampleSet();
	
	/** Add input-output patterns to sample set.
	* @param x input array
	* @param y target values */
	void add(const Vec3Seq& newInputs, const Vec& newTargets);
	
	/** Get input vector at index k. */
	//const Eigen::VectorXd & x(size_t k);
	const Vec3& x(size_t k);
	
	/** Get target value at index k. */
	double y(size_t k);
	/** Set target value at index i. */
	bool set_y(size_t i, double y);
	
	/** Get reference to vector of target values. */
	const Vec& y();
	
	/** Get number of samples. */
	inline size_t size() { return n; };
	
	/** Clear sample set. */
	void clear();
	
	/** Check if sample set is empty. */
	inline bool empty () { return n==0; };

private:
	/** Container holding input vectors. */
	Vec3Seq X;
	/** Container holding target values. */
	Vec Y;
	/** Number of samples. */
	size_t n;
};

//------------------------------------------------------------------------------

}
#endif /* __SAMPLESET_H__ */
