#ifndef NBODY_ENGINE_SIMPLE_H
#define NBODY_ENGINE_SIMPLE_H

#include "nbody_engine.h"

class nbody_engine_simple : public nbody_engine
{
protected:
	class smemory : public memory
	{
		void*	m_data;
		size_t	m_size;
	public:
		smemory( size_t );
		virtual ~smemory();
		void* data() const;
		size_t size() const;
	};

	smemory*			m_mass;
	smemory*			m_y;
	nbody_data*			m_data;
public:
	nbody_engine_simple();
	~nbody_engine_simple();
	virtual const char* type_name() const;
	virtual void init( nbody_data* data );
	virtual void get_data( nbody_data* data );
	virtual size_t problem_size() const;
	virtual memory* y();
	virtual void advise_time( const nbcoord_t& dt );
	virtual nbcoord_t get_time() const;
	virtual void set_time( nbcoord_t t );
	virtual size_t get_step() const;
	virtual void set_step( size_t s );
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff );

	virtual smemory* create_buffer( size_t );
	virtual void free_buffer( memory* );
	virtual void read_buffer( void* dst, memory* src );
	virtual void write_buffer( memory* dst, void* src );
	virtual void copy_buffer( memory* a, const memory* b, size_t aoff, size_t boff );

	virtual void fmadd_inplace( memory* a, const memory* b, const nbcoord_t& c );
	virtual void fmadd( memory* a, const memory* b, const memory* c, const nbcoord_t& d, size_t aoff, size_t boff, size_t coff );
	virtual void fmaddn_inplace( memory* a, const memory* b, const memory* c, size_t bstride, size_t aoff, size_t boff, size_t csize );
	virtual void fmaddn( memory* a, const memory* b, const memory* c, const memory* d, size_t cstride, size_t aoff, size_t boff, size_t coff, size_t dsize );
	virtual void fmaxabs( const memory* a, nbcoord_t& result );
};

#endif // NBODY_ENGINE_SIMPLE_H
