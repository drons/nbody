#include "nbody_solver_stormer.h"
#include "summation.h"

nbody_solver_stormer::nbody_solver_stormer( nbody_data* data ) : nbody_solver( data )
{
}

void nbody_solver_stormer::step( nbcoord_t dt )
{
	nbvertex_t*	vertites = data()->get_vertites();
	nbvertex_t*	velosites = data()->get_velosites();
	size_t		count = data()->get_count();
	bool		first_run( m_dv.empty() );

	if( first_run )
	{
		m_dv.resize( count );
		m_prev_vert.resize( count );
		m_correction_vert.resize( count );
		m_correction_vel.resize( count );
	}

	step_v( vertites, m_dv.data() );

	if( first_run )
	{
		#pragma omp parallel for
		for( size_t n = 0; n < count; ++n )
		{
			velosites[n] = summation_k( velosites[n], m_dv[n]*dt, &m_correction_vel[n] );
			m_prev_vert[n] = vertites[n];
			vertites[n] = summation_k( vertites[n], velosites[n]*dt, &m_correction_vert[n] );
		}
	}
	else
	{
		#pragma omp parallel for
		for( size_t n = 0; n < count; ++n )
		{
			velosites[n] = summation_k( velosites[n], m_dv[n]*dt, &m_correction_vel[n] );
			nbvertex_t	pvert( m_prev_vert[n] );
			nbvertex_t&	cvert( vertites[n] );
			m_prev_vert[n] = cvert;
			cvert = summation_k( cvert, cvert, &m_correction_vert[n] );
			cvert = summation_k( cvert, -pvert, &m_correction_vert[n] );
			cvert = summation_k( cvert, m_dv[n]*dt*dt, &m_correction_vert[n] );
		}
	}
	data()->advise_time( dt );
}

