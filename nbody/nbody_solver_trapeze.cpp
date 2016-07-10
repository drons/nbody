#include "nbody_solver_trapeze.h"
#include "summation.h"

nbody_solver_trapeze::nbody_solver_trapeze() : nbody_solver()
{
	m_f01 = NULL;
	m_predictor = NULL;
	m_coeff = NULL;
}

nbody_solver_trapeze::~nbody_solver_trapeze()
{
	engine()->free( m_f01 );
	engine()->free( m_predictor );
	engine()->free( m_coeff );
}

const char* nbody_solver_trapeze::type_name() const
{
	return "nbody_solver_trapeze";
}

void nbody_solver_trapeze::step( nbcoord_t dt )
{
	const size_t			refine_step_count = 1;
	nbody_engine::memory*	y = engine()->y();
	nbcoord_t				t = engine()->get_time();
	size_t					ps = engine()->problem_size();

	if( m_f01 == NULL )
	{
		m_f01 = engine()->malloc( 2*sizeof(nbcoord_t)*ps );
		m_predictor = engine()->malloc( sizeof(nbcoord_t)*ps );
		m_coeff = engine()->malloc( sizeof(nbcoord_t)*2 );
	}

	engine()->fcompute( t, y, m_f01, 0, 0 );
	engine()->fmadd( m_predictor, y, m_f01, dt, 0, 0, 0 );

	for( size_t s = 0; s <= refine_step_count; ++s )
	{
		engine()->fcompute( t, m_predictor, m_f01, 0, ps );

		nbcoord_t	coeff[] = { dt*0.5, dt*0.5 };
		engine()->memcpy( m_coeff, coeff );

		if( s == refine_step_count )
		{
			engine()->fmaddn( y, m_f01, m_coeff, ps, 0, 0, 2 );
		}
		else
		{
			engine()->fmaddn( m_predictor, y, m_f01, m_coeff, ps, 0, 0, 0, 2 );
		}
	}
	engine()->advise_time( dt );
}
