#ifndef NBODY_ENGINE_H
#define NBODY_ENGINE_H

#include "nbody_data.h"

/*!
	Compute engine
*/
class nbody_engine
{
	size_t	m_compute_count;
public:
	nbody_engine();
	virtual ~nbody_engine();
	virtual void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv ) = 0;

	void advise_compute_count();
	size_t get_compute_count() const;
};

#endif // NBODY_ENGINE_H
