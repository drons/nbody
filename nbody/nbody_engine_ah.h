#ifndef NBODY_ENGINE_AH_H
#define NBODY_ENGINE_AH_H

#include "nbody_engine_simple.h"

/*
	A Numerical Integration Scheme  for the N-Body Gravitational Problem
	A. AHMAD AND L. COHEN 1973

	https://courses.physics.ucsd.edu/2016/Winter/physics141/Lectures/Lecture8/AhmadCohen.pdf
*/
class NBODY_DLL nbody_engine_ah : public nbody_engine_simple
{
	std::vector< std::vector<size_t> >	m_adjacent_body;
	std::vector< nbvertex_t >			m_univerce_force;
	size_t								m_full_recompute_rate;
	nbcoord_t							m_max_dist_sqr;
	nbcoord_t							m_min_force_sqr;
public:
	nbody_engine_ah(size_t full_recompute_rate = 1000, nbcoord_t max_dist = 10, nbcoord_t min_force = 1e-4);
	const char* type_name() const override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
private:
	void fcompute_full(const smemory* y, smemory* f);
	void fcompute_sparse(const smemory* y, smemory* f);
};

#endif // NBODY_ENGINE_SPARSE_H
