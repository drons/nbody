#include "nbody_fcompute_sparse.h"

nbody_fcompute_sparse::nbody_fcompute_sparse()
{
}

void nbody_fcompute_sparse::fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv )
{
	advise_compute_count();

	size_t				count = data->get_count();
	const nbcoord_t*	mass = data->get_mass();
	nbcoord_t			max_dist_sqr = 100;
	nbcoord_t			min_force = 1e-8;

	if( data->get_step() % 1000 == 0 )
	{
		if( m_univerce_force.size() != count )
		{
			m_univerce_force.resize( count );
		}

		if( m_adjacent_body.size() != count )
		{
			m_adjacent_body.resize( count );
		}

		for( size_t n = 0; n != count; ++n )
		{
			m_univerce_force[n] = nbvertex_t(0,0,0);
			m_adjacent_body[n].resize(0);
		}

		#pragma omp parallel for
		for( size_t body1 = 0; body1 < count; ++body1 )
		{
			const nbvertex_t&	v1( vertites[ body1 ] );
			nbvertex_t		total_univerce_force;
			nbvertex_t		total_force;

			for( size_t body2 = 0; body2 != count; ++body2 )
			{
				if( body1 == body2 )
					continue;

				const nbvertex_t&	v2( vertites[ body2 ] );
				const nbvertex_t	f( data->force( v1, v2, mass[body1], mass[body2] ) );

				if( (v1 - v2).norm() < max_dist_sqr ||
					f.norm() > min_force )
				{
					m_adjacent_body[ body1 ].push_back( body2 );
				}
				else
				{
					total_univerce_force += f;
				}
				total_force += f;
			}
			dv[body1] = total_force/mass[body1];
			m_univerce_force[body1] = total_univerce_force;
		}
	}
	else
	{
		#pragma omp parallel for
		for( size_t body1 = 0; body1 < count; ++body1 )
		{
			const nbvertex_t&	v1( vertites[ body1 ] );
			nbvertex_t			total_force( m_univerce_force[body1] );
			size_t*				body2_indites = m_adjacent_body[ body1 ].data();
			size_t				body2_count = m_adjacent_body[ body1 ].size();

			for( size_t idx = 0; idx != body2_count; ++idx )
			{
				size_t				body2( body2_indites[idx] );
				const nbvertex_t&	v2( vertites[ body2 ] );
				const nbvertex_t	f( data->force( v1, v2, mass[body1], mass[body2] ) );
				total_force += f;
			}

			dv[body1] = total_force/mass[body1];
		}
	}
}
