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
	//! Compute f( t, y )
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f ) = 0;

	virtual memory* malloc( size_t ) = 0;
	virtual void free( memory* ) = 0;
	virtual void memcpy( void*, memory* ) = 0;
	virtual void memcpy( memory*, void* ) = 0;

	//! a[i] += b[i]*c
	virtual void fmadd( memory* a, const memory* b, const nbcoord_t& c ) = 0;

	void advise_compute_count();
	size_t get_compute_count() const;
};

#endif // NBODY_ENGINE_H
