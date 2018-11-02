#ifndef NBODY_ENGINE_SIMPLE_BH_H
#define NBODY_ENGINE_SIMPLE_BH_H

#include "nbody_engine_openmp.h"

/*!
  Barnes–Hut simulation
  https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
 */
class NBODY_DLL nbody_engine_simple_bh : public nbody_engine_openmp
{
public:
	enum e_traverse_type
	{
		ett_cycle,
		ett_nested_tree
	};
private:
	nbcoord_t			m_distance_to_node_radius_ratio;
	e_traverse_type		m_traverse_type;
public:
	nbody_engine_simple_bh(nbcoord_t distance_to_node_radius_ratio = 0, e_traverse_type tt = ett_cycle);
	virtual const char* type_name() const override;
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f, size_t yoff, size_t foff) override;
	virtual void print_info() const override;
};

#endif // NBODY_ENGINE_SIMPLE_BH_H
