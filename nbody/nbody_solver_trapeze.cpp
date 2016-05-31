#include "nbody_solver_trapeze.h"
#include "summation.h"

nbody_solver_trapeze::nbody_solver_trapeze( nbody_data* data ) : nbody_solver( data )
{
}

void nbody_solver_trapeze::step( nbcoord_t dt )
{
	nbvertex_t*	vertites = data()->get_vertites();
	nbvertex_t*	velosites = data()->get_velosites();
	size_t		count = data()->get_count();
	size_t		refine_step_count = 1;
	nbcoord_t	dt05 = dt*0.5;

	if( m_dv0.empty() )
	{
		m_dv0.resize( count );
		m_dv1.resize( count );
		m_predictor_vert.resize( count );
		m_predictor_vel.resize( count );
		m_velosites0.resize( count );
	}

	step_v( vertites, m_dv0.data() );
	#pragma omp parallel for
	for( size_t n = 0; n < count; ++n )
	{
		m_velosites0[n] = velosites[n];
		nbvertex_t predictor_vel( velosites[n] + m_dv0[n]*dt );
		m_predictor_vert[n] = vertites[n] + predictor_vel*dt;
	}

	for( size_t s = 0; s <= refine_step_count; ++s )
	{
		step_v( m_predictor_vert.data(), m_dv1.data() );

		if( s == refine_step_count )
		{
			#pragma omp parallel for
			for( size_t n = 0; n < count; ++n )
			{
				velosites[n] += ( m_dv0[n] + m_dv1[n] )*dt05;
				vertites[n] += ( m_velosites0[n] + velosites[n] )*dt05;
			}
		}
		else
		{
			#pragma omp parallel for
			for( size_t n = 0; n < count; ++n )
			{
				nbvertex_t predictor_vel( velosites[n] + ( m_dv0[n] + m_dv1[n] )*dt05 );
				m_predictor_vert[n] = vertites[n] + ( m_velosites0[n] + predictor_vel )*dt05;
			}
		}
	}
	data()->advise_time( dt );
}
