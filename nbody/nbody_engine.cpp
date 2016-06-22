#include "nbody_engine.h"

nbody_engine::nbody_engine()
{
	m_compute_count = 0;
}

nbody_engine::~nbody_engine()
{

}

void nbody_engine::advise_compute_count()
{
	m_compute_count++;
}

size_t nbody_engine::get_compute_count() const
{
	return m_compute_count;
}
