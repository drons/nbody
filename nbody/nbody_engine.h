#ifndef NBODY_ENGINE_H
#define NBODY_ENGINE_H

#include "nbody_data.h"

//! ODE order
enum e_ode_order
{
	eode_first_order = 1,	// y' = f(t, y)
	eode_second_order = 1,	// y" = f(t, y)
};

/*!
	Compute engine for ODE y' = f(t, y) or y" = f(t, y)
*/
class NBODY_DLL nbody_engine
{
	size_t	m_compute_count;
	e_ode_order	m_ode_order;
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
	typedef std::vector<memory*>	memory_array;

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
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) = 0;

	virtual memory* create_buffer(size_t) = 0;
	virtual void free_buffer(memory*) = 0;
	virtual memory_array create_buffers(size_t size, size_t count);
	virtual void free_buffers(memory_array&);
	virtual void read_buffer(void* dst, const memory* src) = 0;
	virtual void write_buffer(memory* dst, const void* src) = 0;
	//! a[i] = b[i]
	virtual void copy_buffer(memory* a, const memory* b) = 0;
	//! a[i] = value
	virtual void fill_buffer(memory* a, const nbcoord_t& value) = 0;
	//! a[i] += b[i]*c
	virtual void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) = 0;
	//! a[i] = b[i] + c[i]*d
	virtual void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d) = 0;
	//! a[i] += sum( b[k][i]*c[k], k=[0...b.size()) )
	virtual void fmaddn_inplace(memory* a, const memory_array& b, const nbcoord_t* c);
	//! a[i] += sum( b[k][i]*c[k], k=[0...b.size()) ) Kahan summation with correction
	virtual void fmaddn_corr(memory* a, memory* corr, const memory_array& b, const nbcoord_t* c);
	//! a[i] = b[i] + sum( c[k][i]*d[k], k=[0...c.size()) )
	virtual void fmaddn(memory* a, const memory* b, const memory_array& c,
						const nbcoord_t* d, size_t dsize);
	//! @result = max( fabs(a[k]), k=[0...asize) )
	virtual void fmaxabs(const memory* a, nbcoord_t& result) = 0;
	//! Print engine info
	virtual void print_info() const;

	void advise_compute_count();
	size_t get_compute_count() const;
	//! Set engine's ODE order
	void set_ode_order(e_ode_order);
	//! ODE order
	e_ode_order get_ode_order() const;
};

#endif // NBODY_ENGINE_H
