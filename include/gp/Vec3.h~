/** @file Vec3.h
 * 
 * Mathematical routines extract from Golem (Marek Kopicki).
 * 
 * @author	Claudio Zito
 *
 * @copyright  Copyright (C) 2015 Claudio Zito & Marek Kopicki, University of Birmingham, UK
 *
 * @license  This file copy is licensed to you under the terms described in
 *           the License.txt file included in this distribution.
 *
 */

#ifndef _GP_VEC3_H_
#define _GP_VEC3_H_

//------------------------------------------------------------------------------

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <float.h>
#include <algorithm>
#include <assert.h> 

//------------------------------------------------------------------------------

namespace gp {

//------------------------------------------------------------------------------

/** Some utility stdematical functions */

typedef float F32;
typedef double F64;

/** Numeric constants base class for integer numbers */
template <typename Type> class numeric_const {
public:
	/** Integer zero */
	static const Type ZERO;
	/** Integer one */
	static const Type ONE;
	/** Integer two */
	static const Type TWO;
	/** Integer pi */
	static const Type PI;
	/** Integer two pi */
	static const Type TWO_PI;
};
template <typename Type> const Type numeric_const<Type>::ZERO = Type(+0.0);
template <typename Type> const Type numeric_const<Type>::ONE = Type(+1.0);
template <typename Type> const Type numeric_const<Type>::TWO = Type(+2.0);
template <typename Type> const Type numeric_const<Type>::PI = Type(3.14159265358979323846);
template <typename Type> const Type numeric_const<Type>::TWO_PI = Type(6.28318530717958647692);

template <typename Type> inline static bool equals(Type a, Type b, Type eps) {
	return (std::abs(a - b) < eps);
}

//------------------------------------------------------------------------------

/** 3 Element vector class.
*/
template <typename _Real> class _Vec3 {
public:
	/** Real */
	typedef _Real Real;

	/** vector components */
	union {
		struct {
			Real x, y, z;
		};
		struct {
			Real v1, v2, v3;
		};
		Real v[3];
	};

	/** Default constructor does not do any initialisation.
	*/
	inline _Vec3() {}

	/**	Assigns scalar parameters to all elements.
	*	@param	a		Value to assign to elements.
	*/
	inline _Vec3(F32 a) : v1((Real)a), v2((Real)a), v3((Real)a) {}

	/**	Assigns scalar parameters to all elements.
	*	@param	a		Value to assign to elements.
	*/
	inline _Vec3(F64 a) : v1((Real)a), v2((Real)a), v3((Real)a) {}

	/** Initialises from 3 scalar parameters.
	*	@param	v1		Value to initialise v1 component.
	*	@param	v2		Value to initialise v1 component.
	*	@param	v3		Value to initialise v1 component.
	*/
	inline _Vec3(F32 v1, F32 v2, F32 v3) : v1((Real)v1), v2((Real)v2), v3((Real)v3) {}

	/** Initialises from 3 scalar parameters.
	*	@param	v1		Value to initialise v1 component.
	*	@param	v2		Value to initialise v1 component.
	*	@param	v3		Value to initialise v1 component.
	*/
	inline _Vec3(F64 v1, F64 v2, F64 v3) : v1((Real)v1), v2((Real)v2), v3((Real)v3) {}

	/**	Copy constructor.
	*/
	inline _Vec3(const _Vec3<F32> &v) : v1((Real)v.v1), v2((Real)v.v2), v3((Real)v.v3) {}

	/**	Copy constructor.
	*/
	inline _Vec3(const _Vec3<F64> &v) : v1((Real)v.v1), v2((Real)v.v2), v3((Real)v.v3) {}

	/** Static initialisation member - zero.
	*/
	static inline _Vec3 zero() {
		return _Vec3(numeric_const<Real>::ZERO);
	}

	/** returns true if the object is valid
	*/
	inline bool isValid() const {
		return isFinite();
	}

	/** the default configuration
	*/
	inline void setToDefault() {
		setZero();
	}

	/**	Access the data as an array.
	*/
	inline Real* get() {
		return v;
	}
	inline const Real *get() const {
		return v;
	}

	/** Writes the 3 values to v.
	*/
	inline void get(F32* v) const {
		v[0] = (F32)this->v1;
		v[1] = (F32)this->v2;
		v[2] = (F32)this->v3;
	}

	/** Writes the 3 values to v.
	*/
	inline void get(F64* v) const {
		v[0] = (F64)this->v1;
		v[1] = (F64)this->v2;
		v[2] = (F64)this->v3;
	}

	inline void set(Real a) {
		v1 = a;
		v2 = a;
		v3 = a;
	}

	inline void set(Real v1, Real v2, Real v3) {
		this->v1 = v1;
		this->v2 = v2;
		this->v3 = v3;
	}
	
	/** reads 3 consecutive values from the ptr passed
	*/
	inline void  set(const F32* v) {
		v1 = (Real)v[0];
		v2 = (Real)v[1];
		v3 = (Real)v[2];
	}

	/** reads 3 consecutive values from the ptr passed
	*/
	inline void set(const F64* v) {
		v1 = (Real)v[0];
		v2 = (Real)v[1];
		v3 = (Real)v[2];
	}
	
	/** this = v
	*/
	inline void  set(const _Vec3& v) {
		v1 = v.v1;
		v2 = v.v2;
		v3 = v.v3;
	}

	/** this = 0
	*/
	inline void setZero() {
		v1 = numeric_const<Real>::ZERO;
		v2 = numeric_const<Real>::ZERO;
		v3 = numeric_const<Real>::ZERO;
	}
	
	/** this = -a
	*/
	inline void  setNegative(const _Vec3& v) {
		v1 = -v.v1;
		v2 = -v.v2;
		v3 = -v.v3;
	}

	/** this = -this
	*/
	inline void  setNegative() {
		v1 = -v1;
		v2 = -v2;
		v3 = -v3;
	}

	/** sets the vector's magnitude
	*/
	inline void setMagnitude(Real length) {
		const Real m = magnitude();

		if (std::abs(m) > numeric_const<Real>::ZERO) {
			const Real newLength = length / m;
			v1 *= newLength;
			v2 *= newLength;
			v3 *= newLength;
		}
	}

	/** normalises the vector
	*/
	inline Real normalise() {
		const Real m = magnitude();
		
		if (std::abs(m) > numeric_const<Real>::ZERO) {
			const Real length = numeric_const<Real>::ONE / m;
			v1 *= length;
			v2 *= length;
			v3 *= length;
		}
		
		return m;
	}

	/** this = element wise min(this,other)
	*/
	inline void min(const _Vec3& v) {
		if (v1 > v.v1) v1 = v.v1;
		if (v2 > v.v2) v2 = v.v2;
		if (v3 > v.v3) v3 = v.v3;
	}
	/** this = element wise max(this,other)
	*/
	inline void max(const _Vec3& v) {
		if (v1 < v.v1) v1 = v.v1;
		if (v2 < v.v2) v2 = v.v2;
		if (v3 < v.v3) v3 = v.v3;
	}

	/** this = a + b
	*/
	inline void add(const _Vec3& a, const _Vec3& b) {
		v1 = a.v1 + b.v1;
		v2 = a.v2 + b.v2;
		v3 = a.v3 + b.v3;
	}

	/** this = a - b
	*/
	inline void subtract(const _Vec3& a, const _Vec3& b) {
		v1 = a.v1 - b.v1;
		v2 = a.v2 - b.v2;
		v3 = a.v3 - b.v3;
	}

	/** this = s * a;
	*/
	inline void multiply(Real s, const _Vec3& a) {
		v1 = a.v1 * s;
		v2 = a.v2 * s;
		v3 = a.v3 * s;
	}

	/** this[i] = a[i] * b[i], for all i.
	*/
	inline void arrayMultiply(const _Vec3& a, const _Vec3& b) {
		v1 = a.v1 * b.v1;
		v2 = a.v2 * b.v2;
		v3 = a.v3 * b.v3;
	}

	/** this = s * a + b;
	*/
	inline void multiplyAdd(Real s, const _Vec3& a, const _Vec3& b) {
		v1 = s * a.v1 + b.v1;
		v2 = s * a.v2 + b.v2;
		v3 = s * a.v3 + b.v3;
	}

	/** this = s * a + t * b;
	*/
	inline void linear(Real s, const _Vec3& a, Real t, const _Vec3& b) {
		v1 = s * a.v1 + t * b.v1;
		v2 = s * a.v2 + t * b.v2;
		v3 = s * a.v3 + t * b.v3;
	}

	/** this = a + s * (b - a);
	*/
	inline void interpolate(const _Vec3& a, const _Vec3& b, Real s) {
		v1 = a.v1 + s * (b.v1 - a.v1);
		v2 = a.v2 + s * (b.v2 - a.v2);
		v3 = a.v3 + s * (b.v3 - a.v3);
	}

	/** returns the magnitude
	*/
	inline Real magnitude() const {
		return std::sqrt(magnitudeSqr());
	}

	/** returns the squared magnitude
	*/
	inline Real magnitudeSqr() const {
		return v1 * v1 + v2 * v2 + v3 * v3;
	}

	/** returns (this - other).distance();
	*/
	inline Real distance(const _Vec3& v) const {
		return std::sqrt(distanceSqr(v));
	}

	/** returns (this - other).distanceSqr();
	*/
	inline Real distanceSqr(const _Vec3& v) const {
		const Real dv1 = v1 - v.v1;
		const Real dv2 = v2 - v.v2;
		const Real dv3 = v3 - v.v3;

		return dv1 * dv1 + dv2 * dv2 + dv3 * dv3;
	}

	/** returns the dot/scalar product of this and other.
	*/
	inline Real dot(const _Vec3& v) const {
		return v1 * v.v1 + v2 * v.v2 + v3 * v.v3;
	}

	/** cross product, this = left v1 right
	*/
	inline void cross(const _Vec3& left, const _Vec3& right) {
		// temps needed in case left or right is this.
		const Real a = (left.v2 * right.v3) - (left.v3 * right.v2);
		const Real b = (left.v3 * right.v1) - (left.v1 * right.v3);
		const Real c = (left.v1 * right.v2) - (left.v2 * right.v1);

		v1 = a;
		v2 = b;
		v3 = c;
	}
	/** cross product
	*/
	_Vec3 cross(const _Vec3& v) const {
		return _Vec3(v2*v.v3 - v3*v.v2, v3*v.v1 - v1*v.v3, v1*v.v2 - v2*v.v1);
	}

	/** Generates uniform random direction
	*/
//	template <typename Rand> void next(const Rand &rand) {
//		const Real phi = numeric_const<Real>::TWO_PI * rand.template nextUniform<Real>();
//		const Real cos = numeric_const<Real>::TWO*rand.template nextUniform<Real>() - numeric_const<Real>::ONE;
//		const Real sin = std::sqrt(numeric_const<Real>::ONE - cos*cos);
//		set(cos, sin * std::cos(phi), sin * std::sin(phi));
//	}

	/** tests for exact zero vector
	*/
	inline bool isZero() const {
		return v1 == numeric_const<Real>::ZERO && v2 == numeric_const<Real>::ZERO && v3 == numeric_const<Real>::ZERO;
	}

	/** tests for positive vector
	*/
	inline bool isPositive() const {
		return v1 > numeric_const<Real>::ZERO && v2 > numeric_const<Real>::ZERO && v3 > numeric_const<Real>::ZERO;
	}

	/** tests for negative vector
	*/
	inline bool isNegative() const {
		return v1 < numeric_const<Real>::ZERO && v2 < numeric_const<Real>::ZERO && v3 < numeric_const<Real>::ZERO;
	}

	/** tests for finite vector
	*/
	inline bool isFinite() const {
		return std::isfinite(v1) && std::isfinite(v2) && std::isfinite(v3);
	}

	/** returns true if this and arg's elems are within epsilon of each other.
	*/
	inline bool equals(const _Vec3& v, Real epsilon) const {
		return
			equals(v1, v.v1, epsilon) &&
			equals(v2, v.v2, epsilon) &&
			equals(v3, v.v3, epsilon);
	}

	/**	Assignment operator.
	*/
	inline const _Vec3& operator = (const _Vec3& v) {
		v1 = v.v1;	v2 = v.v2;	v3 = v.v3;
		return *this;
	}

	/** Access the data as an array.
	*	@param	idx	Array index.
	*	@return		Array element pointed by idx.
	*/
	inline Real& operator [] (size_t idx) {
		assert(idx <= 2);
		return v[idx];
	}
	inline const Real& operator [] (size_t idx) const {
		assert(idx <= 2);
		return v[idx];
	}
	
	/** true if all the members are smaller.
	*/
	inline bool operator < (const _Vec3& v) const {
		return (v1 < v.v1) && (v2 < v.v2) && (v3 < v.v3);
	}

	/** true if all the members are smaller or equal.
	*/
	inline bool operator <= (const _Vec3& v) const {
		return (v1 <= v.v1) && (v2 <= v.v2) && (v3 <= v.v3);
	}

	/** true if all the members are larger.
	*/
	inline bool operator > (const _Vec3& v) const {
		return (v1 > v.v1) && (v2 > v.v2) && (v3 > v.v3);
	}

	/** true if all the members are larger or equal.
	*/
	inline bool operator >= (const _Vec3& v) const {
		return (v1 >= v.v1) && (v2 >= v.v2) && (v3 >= v.v3);
	}

	/** returns true if the two vectors are exactly equal.
	*/
	inline bool operator == (const _Vec3& v) const {
		return (v1 == v.v1) && (v2 == v.v2) && (v3 == v.v3);
	}

	/** returns true if the two vectors are exactly unequal.
	*/
	inline bool operator != (const _Vec3& v) const {
		return (v1 != v.v1) || (v2 != v.v2) || (v3 != v.v3);
	}

	/** negation
	*/
	_Vec3 operator - () const {
		return _Vec3(-v1, -v2, -v3);
	}
	/** vector addition
	*/
	_Vec3 operator + (const _Vec3 & v) const {
		return _Vec3(v1 + v.v1, v2 + v.v2, v3 + v.v3);
	}
	/** vector difference
	*/
	_Vec3 operator - (const _Vec3 & v) const {
		return _Vec3(v1 - v.v1, v2 - v.v2, v3 - v.v3);
	}
	/** scalar post-multiplication
	*/
	_Vec3 operator * (Real f) const {
		return _Vec3(v1 * f, v2 * f, v3 * f);
	}
	/** scalar division
	*/
	_Vec3 operator / (Real f) const {
		f = Real(1.0) / f;
		return _Vec3(v1 * f, v2 * f, v3 * f);
	}
	/** vector addition
	*/
	_Vec3& operator += (const _Vec3& v) {
		v1 += v.v1;
		v2 += v.v2;
		v3 += v.v3;
		return *this;
	}
	/** vector difference
	*/
	_Vec3& operator -= (const _Vec3& v) {
		v1 -= v.v1;
		v2 -= v.v2;
		v3 -= v.v3;
		return *this;
	}
	/** scalar multiplication
	*/
	_Vec3& operator *= (Real f) {
		v1 *= f;
		v2 *= f;
		v3 *= f;
		return *this;
	}
	/** scalar division
	*/
	_Vec3& operator /= (Real f) {
		f = Real(1.0) / f;
		v1 *= f;
		v2 *= f;
		v3 *= f;
		return *this;
	}
	/** cross product
	*/
	_Vec3 operator ^ (const _Vec3& v) const {
		return _Vec3(v2*v.v3 - v3*v.v2, v3*v.v1 - v1*v.v3, v1*v.v2 - v2*v.v1);
	}
	/** dot product
	*/
	Real operator | (const _Vec3& v) const {
		return v1 * v.v1 + v2 * v.v2 + v3 * v.v3;
	}
};

//------------------------------------------------------------------------------

/** Default type */
typedef _Vec3<F64> Vec3;
typedef std::vector<F64> Vec;
typedef std::vector<Vec3> Vec3Seq;

//------------------------------------------------------------------------------

};	// namespace

#endif /*_GOLEM_std_VEC3_H_*/
