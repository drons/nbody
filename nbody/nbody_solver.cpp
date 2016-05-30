#include "nbody_solver.h"
#include "summation.h"
#include <QDebug>

nbody_solver::nbody_solver( nbody_data* data )
	: m_data( data )
{
}

nbody_solver::~nbody_solver()
{

}

nbody_data* nbody_solver::data() const
{
	return m_data;
}

void nbody_solver::step_v( const nbvertex_t* vertites, nbvertex_t* dv )
{
	size_t				count = m_data->get_count();
	const nbcoord_t*	mass = m_data->get_mass();
	const size_t		block = NBODY_DATA_BLOCK_SIZE;

	#pragma omp parallel for
	for( size_t n1 = 0; n1 < count; n1 += block )
	{
		const nbvertex_t*	v1( vertites + n1 );
		nbcoord_t			x1[block];
		nbcoord_t			y1[block];
		nbcoord_t			z1[block];
		nbcoord_t			total_force_x[block];
		nbcoord_t			total_force_y[block];
		nbcoord_t			total_force_z[block];

		for( size_t b1 = 0; b1 != block; ++b1 )
		{
			x1[b1] = v1[b1].x;
			y1[b1] = v1[b1].y;
			z1[b1] = v1[b1].z;
			total_force_x[b1] = 0;
			total_force_y[b1] = 0;
			total_force_z[b1] = 0;
		}
		for( size_t n2 = 0; n2 < count; n2 += block )
		{
			const nbvertex_t*	v2( vertites + n2 );
			nbcoord_t			x2[block];
			nbcoord_t			y2[block];
			nbcoord_t			z2[block];
			nbcoord_t			m2[block];

			for( size_t b2 = 0; b2 != block; ++b2 )
			{
				x2[b2] = v2[b2].x;
				y2[b2] = v2[b2].y;
				z2[b2] = v2[b2].z;
				m2[b2] = mass[n2 + b2];
			}

			for( size_t b1 = 0; b1 != block; ++b1 )
			{
				for( size_t b2 = 0; b2 != block; ++b2 )
				{
					nbcoord_t		dx = x1[b1] - x2[b2];
					nbcoord_t		dy = y1[b1] - y2[b2];
					nbcoord_t		dz = z1[b1] - z2[b2];
					nbcoord_t		r2( dx*dx + dy*dy + dz*dz );
					if( r2 < NBODY_MIN_R )
						r2 = NBODY_MIN_R;
					nbcoord_t		r = sqrt( r2 );
					nbcoord_t		coeff = (m2[b2])/(r*r2);

					dx *= coeff;
					dy *= coeff;
					dz *= coeff;

					total_force_x[b1] -= dx;
					total_force_y[b1] -= dy;
					total_force_z[b1] -= dz;
				}
			}
		}

		for( size_t b1 = 0; b1 != block; ++b1 )
		{
			dv[n1 + b1] = nbvertex_t( total_force_x[b1], total_force_y[b1], total_force_z[b1] );
		}
	}
}

void nbody_solver::step_v2( const nbvertex_t* vertites, nbvertex_t* dv )
{
	size_t				count = m_data->get_count();
	const nbcoord_t*	mass = m_data->get_mass();

	#pragma omp parallel for
	for( size_t body1 = 0; body1 < count; ++body1 )
	{
		const nbvertex_t&	v1( vertites[ body1 ] );
		nbvertex_t			total_force;
		for( size_t body2 = 0; body2 != count; ++body2 )
		{
			if( body1 == body2 )
				continue;
			const nbvertex_t&	v2( vertites[ body2 ] );
			const nbvertex_t	f( m_data->force( v1, v2, mass[body1], mass[body2] ) );
			total_force += f;
		}

		dv[body1] = total_force/mass[body1];
	}
}

void nbody_solver::step_v0( const nbvertex_t* vertites, nbvertex_t* dv )
{
	size_t				count = m_data->get_count();
	const nbcoord_t*	mass = m_data->get_mass();
	nbcoord_t			max_dist_sqr = 100;
	nbcoord_t			min_force = 1e-8;

	if( m_data->get_step() % 1000 == 0 )
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
				const nbvertex_t	f( m_data->force( v1, v2, mass[body1], mass[body2] ) );

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
				const nbvertex_t	f( m_data->force( v1, v2, mass[body1], mass[body2] ) );
				total_force += f;
			}

			dv[body1] = total_force/mass[body1];
		}
	}
}
