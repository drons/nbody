#include "nbody_engine_ah.h"

nbody_engine_ah::nbody_engine_ah(size_t full_recompute_rate,
								 nbcoord_t max_dist, nbcoord_t min_force) :
	m_full_recompute_rate(full_recompute_rate),
	m_max_dist_sqr(max_dist * max_dist),
	m_min_force_sqr(min_force * min_force)
{
}

const char* nbody_engine_ah::type_name() const
{
	return "nbody_engine_ah";
}

void nbody_engine_ah::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
	smemory*		f = dynamic_cast<smemory*>(_f);

	if(y == NULL)
	{
		qDebug() << "y is not smemory";
		return;
	}
	if(f == NULL)
	{
		qDebug() << "f is not smemory";
		return;
	}

	if(m_data->get_step() % m_full_recompute_rate == 0)
	{
		fcompute_full(y, f);
	}
	else
	{
		fcompute_sparse(y, f);
	}
}

void nbody_engine_ah::fcompute_full(const nbody_engine_simple::smemory* y, nbody_engine_simple::smemory* f)
{
	advise_compute_count();

	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = reinterpret_cast<const nbcoord_t*>(y->data());
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	vx = rx + 3 * count;
	const nbcoord_t*	vy = rx + 4 * count;
	const nbcoord_t*	vz = rx + 5 * count;

	nbcoord_t*			frx = reinterpret_cast<nbcoord_t*>(f->data());
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2 * count;
	nbcoord_t*			fvx = frx + 3 * count;
	nbcoord_t*			fvy = frx + 4 * count;
	nbcoord_t*			fvz = frx + 5 * count;
	const nbcoord_t*	mass = reinterpret_cast<const nbcoord_t*>(m_mass->data());

	if(m_univerce_force.size() != count)
	{
		m_univerce_force.resize(count);
	}

	if(m_adjacent_body.size() != count)
	{
		m_adjacent_body.resize(count);
	}

	for(size_t n = 0; n != count; ++n)
	{
		m_univerce_force[n] = nbvertex_t(0, 0, 0);
		m_adjacent_body[n].resize(0);
	}

	#pragma omp parallel for
	for(size_t body1 = 0; body1 < count; ++body1)
	{
		const nbvertex_t	v1(rx[ body1 ], ry[ body1 ], rz[ body1 ]);
		nbvertex_t			total_univerce_force;
		nbvertex_t			total_force;

		for(size_t body2 = 0; body2 != count; ++body2)
		{
			if(body1 == body2)
			{
				continue;
			}

			const nbvertex_t	v2(rx[ body2 ], ry[ body2 ], rz[ body2 ]);
			const nbvertex_t	force(m_data->force(v1, v2, mass[body1], mass[body2]));

			if((v1 - v2).norm() < m_max_dist_sqr ||
			   force.norm() > m_min_force_sqr)
			{
				m_adjacent_body[ body1 ].push_back(body2);
			}
			else
			{
				total_univerce_force += force;
			}
			total_force += force;
		}
		frx[body1] = vx[body1];
		fry[body1] = vy[body1];
		frz[body1] = vz[body1];
		fvx[body1] = total_force.x / mass[body1];
		fvy[body1] = total_force.y / mass[body1];
		fvz[body1] = total_force.z / mass[body1];
		m_univerce_force[body1] = total_univerce_force;
	}
}

void nbody_engine_ah::fcompute_sparse(const nbody_engine_simple::smemory* y, nbody_engine_simple::smemory* f)
{
	advise_compute_count();

	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = reinterpret_cast<const nbcoord_t*>(y->data());
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	vx = rx + 3 * count;
	const nbcoord_t*	vy = rx + 4 * count;
	const nbcoord_t*	vz = rx + 5 * count;

	nbcoord_t*			frx = reinterpret_cast<nbcoord_t*>(f->data());
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2 * count;
	nbcoord_t*			fvx = frx + 3 * count;
	nbcoord_t*			fvy = frx + 4 * count;
	nbcoord_t*			fvz = frx + 5 * count;
	const nbcoord_t*	mass = reinterpret_cast<const nbcoord_t*>(m_mass->data());

	#pragma omp parallel for
	for(size_t body1 = 0; body1 < count; ++body1)
	{
		const nbvertex_t	v1(rx[ body1 ], ry[ body1 ], rz[ body1 ]);
		nbvertex_t			total_force(m_univerce_force[body1]);
		size_t*				body2_indites = m_adjacent_body[ body1 ].data();
		size_t				body2_count = m_adjacent_body[ body1 ].size();

		for(size_t idx = 0; idx != body2_count; ++idx)
		{
			size_t				body2(body2_indites[idx]);
			const nbvertex_t	v2(rx[ body2 ], ry[ body2 ], rz[ body2 ]);
			const nbvertex_t	force(m_data->force(v1, v2, mass[body1], mass[body2]));
			total_force += force;
		}

		frx[body1] = vx[body1];
		fry[body1] = vy[body1];
		frz[body1] = vz[body1];
		fvx[body1] = total_force.x / mass[body1];
		fvy[body1] = total_force.y / mass[body1];
		fvz[body1] = total_force.z / mass[body1];
	}
}
