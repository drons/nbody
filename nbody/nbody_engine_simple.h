#ifndef NBODY_ENGINE_SIMPLE_H
#define NBODY_ENGINE_SIMPLE_H

#include "nbody_engine.h"

class nbody_engine_simple : public nbody_engine
{
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
	virtual void init( nbody_data* data );
	virtual void get_data( nbody_data* data );
	virtual size_t problem_size() const;
	virtual memory* y();
	virtual void advise_time( const nbcoord_t& dt );
	virtual void fcompute( const nbcoord_t& t, const memory* y, memory* f );

	virtual smemory* malloc( size_t );
	virtual void free( memory* );
	virtual void memcpy( void* src, memory* dst );
	virtual void memcpy( memory*, void* );

	virtual void fmadd( memory* a, const memory* b, const nbcoord_t& c );

};

#endif // NBODY_ENGINE_SIMPLE_H
