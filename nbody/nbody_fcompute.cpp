#include "nbody_fcompute.h"

nbody_fcompute::nbody_fcompute()
{
	m_compute_count = 0;
}

nbody_fcompute::~nbody_fcompute()
{

}

void nbody_fcompute::advise_compute_count()
{
	m_compute_count++;
}

size_t nbody_fcompute::get_compute_count() const
{
	return m_compute_count;
}
