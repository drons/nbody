#ifndef NBODY_ENGINE_H
#define NBODY_ENGINE_H

#include "nbody_data.h"

/*!
	Compute engine for ODE y' = f(t, y)
*/
class nbody_engine
{
	size_t	m_compute_count;
public:
	class memory
	{
	public:
		memory();
		virtual ~memory();
	};
public:
	nbody_engine();
	virtual ~nbody_engine();
	//! Initialize engine
	virtual void init( nbody_data* data ) = 0;
	//! Load data from engine to nbody_data
	virtual void get_data( nbody_data* data ) = 0;
	//! @returns vector <y> size
	virtual size_t problem_size() const = 0;
	//! Current <y> state
	virtual memory* y() = 0;
	//! Advise time state
	virtual void advise_time( const nbcoord_t& dt ) = 0;
	//! Get current time
	virtual nbcoord_t get_time() const = 0;
	//! Get current step
	virtual size_t get_step() const = 0;
	//! Compute f( t, y )
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff ) = 0;

	virtual memory* malloc( size_t ) = 0;
	virtual void free( memory* ) = 0;
	virtual void memcpy( void* dst, memory* src ) = 0;
	virtual void memcpy( memory* dst, void* src ) = 0;
	//! a[i+aoff] = b[i+boff]
	virtual void memcpy( memory* a, const memory* b, size_t aoff, size_t boff ) = 0;

	//! a[i] += b[i]*c
	virtual void fmadd( memory* a, const memory* b, const nbcoord_t& c ) = 0;
	//! a[i+aoff] = b[i+boff] + c[i+coff]*d
	virtual void fmadd( memory* a, const memory* b, const memory* c, const nbcoord_t& d, size_t aoff, size_t boff, size_t coff ) = 0;
	//! a[i+aoff] += sum( b[i+boff+k*bstride]*c[k], k=[0...csize) )
	virtual void fmaddn( memory* a, const memory* b, const memory* c, size_t bstride, size_t aoff, size_t boff, size_t csize ) = 0;
	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	virtual void fmaddn( memory* a, const memory* b, const memory* c, const memory* d, size_t cstride, size_t aoff, size_t boff, size_t coff, size_t dsize ) = 0;
	//! @result = max( fabs(a[k]), k=[0...asize) )
	virtual void fmaxabs( const memory* a, nbcoord_t& result ) = 0;


	void advise_compute_count();
	size_t get_compute_count() const;
};

#endif // NBODY_ENGINE_H
