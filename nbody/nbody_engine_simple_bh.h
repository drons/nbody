#ifndef NBODY_ENGINE_SIMPLE_BH_H
#define NBODY_ENGINE_SIMPLE_BH_H

#include "nbody_engine_simple.h"

/*!
  Barnes–Hut simulation
  https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
 */
class nbody_engine_simple_bh : public nbody_engine_simple
{
	nbcoord_t	m_distance_to_node_radius_ratio;
public:
	nbody_engine_simple_bh(nbcoord_t distance_to_node_radius_ratio = 0);
	virtual const char* type_name() const override;
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff) override;
};

#endif // NBODY_ENGINE_SIMPLE_BH_H
