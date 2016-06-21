#ifndef NBODY_FCOMPUTE_H
#define NBODY_FCOMPUTE_H

#include "nbody_data.h"

class nbody_fcompute
{
	size_t	m_compute_count;
public:
	nbody_fcompute();
	virtual ~nbody_fcompute();
	virtual void fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv ) = 0;

	void advise_compute_count();
	size_t get_compute_count() const;
};

#endif // NBODY_FCOMPUTE_H
