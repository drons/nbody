#include "nbody_solver_euler.h"
#include "summation.h"

nbody_solver_euler::nbody_solver_euler( nbody_data* data ) : nbody_solver( data )
{
}

void nbody_solver_euler::step( nbcoord_t dt )
{
	nbvertex_t*	vertites = data()->get_vertites();
	nbvertex_t*	velosites = data()->get_velosites();
	size_t		count = data()->get_count();

	if( m_dv.empty() )
	{
		m_dv.resize( count );
		m_correction_vert.resize( count );
		m_correction_vel.resize( count );
	}

	step_v( vertites, m_dv.data() );

	#pragma omp parallel for
	for( size_t n = 0; n < count; ++n )
	{
		velosites[n] = summation_k( velosites[n], m_dv[n]*dt, &m_correction_vel[n] );
		vertites[n] = summation_k( vertites[n], velosites[n]*dt, &m_correction_vert[n] );
	}
	data()->advise_time( dt );
}
