#ifndef NBODY_ENGINE_H
#define NBODY_ENGINE_H

#include "nbody_data.h"

/*!
	Compute engine for ODE y' = f(t, y)
*/
class NBODY_DLL nbody_engine
{
	size_t	m_compute_count;
	nbody_engine(const nbody_engine&) = delete;
	nbody_engine& operator = (const nbody_engine&) = delete;
public:
	class NBODY_DLL memory
	{
	public:
		memory();
		//! Memory size in bytes
		virtual size_t size() const = 0;
		virtual ~memory();
	};
public:
	nbody_engine();
	virtual ~nbody_engine();
	//! Engine's type name
	virtual const char* type_name() const = 0;
	//! Initialize engine
	virtual void init(nbody_data* data) = 0;
	//! Load data from engine to nbody_data
	virtual void get_data(nbody_data* data) = 0;
	//! @returns vector <y> size
	virtual size_t problem_size() const = 0;
	//! Current <y> state
	virtual memory* get_y() = 0;
	//! Advise time state
	virtual void advise_time(const nbcoord_t& dt) = 0;
	//! Get current time
	virtual nbcoord_t get_time() const = 0;
	//! Set current time
	virtual void set_time(nbcoord_t t) = 0;
	//! Get current step
	virtual size_t get_step() const = 0;
	//! Set current step
	virtual void set_step(size_t s) = 0;
	//! Compute f( t, y )
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff) = 0;

	virtual memory* create_buffer(size_t) = 0;
	virtual void free_buffer(memory*) = 0;
	virtual void read_buffer(void* dst, memory* src) = 0;
	virtual void write_buffer(memory* dst, void* src) = 0;
	//! a[i+aoff] = b[i+boff]
	virtual void copy_buffer(memory* a, const memory* b, size_t aoff, size_t boff) = 0;

	//! a[i] += b[i]*c
	virtual void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) = 0;
	//! a[i+aoff] = b[i+boff] + c[i+coff]*d
	virtual void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d, size_t aoff, size_t boff,
					   size_t coff) = 0;
	//! a[i+aoff] += sum( b[i+boff+k*bstride]*c[k], k=[0...csize) )
	virtual void fmaddn_inplace(memory* a, const memory* b, const memory* c, size_t bstride, size_t aoff, size_t boff,
								size_t csize) = 0;
	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	virtual void fmaddn(memory* a, const memory* b, const memory* c, const memory* d, size_t cstride, size_t aoff,
						size_t boff, size_t coff, size_t dsize) = 0;
	//! @result = max( fabs(a[k]), k=[0...asize) )
	virtual void fmaxabs(const memory* a, nbcoord_t& result) = 0;
	//! Print engine info
	virtual void print_info() const;

	void advise_compute_count();
	size_t get_compute_count() const;
};

#endif // NBODY_ENGINE_H
