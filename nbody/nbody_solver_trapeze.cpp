#include "nbody_solver_trapeze.h"
#include "summation.h"

nbody_solver_trapeze::nbody_solver_trapeze() : nbody_solver()
{
	m_f01 = NULL;
	m_predictor = NULL;
	m_coeff = NULL;
	m_refine_steps_count = 1;
}

nbody_solver_trapeze::~nbody_solver_trapeze()
{
	engine()->free_buffer( m_f01 );
	engine()->free_buffer( m_predictor );
	engine()->free_buffer( m_coeff );
}

const char* nbody_solver_trapeze::type_name() const
{
	return "nbody_solver_trapeze";
}

void nbody_solver_trapeze::set_refine_steps_count( size_t v )
{
	m_refine_steps_count = v;
}

void nbody_solver_trapeze::step( nbcoord_t dt )
{
	nbody_engine::memory*	y = engine()->y();
	nbcoord_t				t = engine()->get_time();
	size_t					ps = engine()->problem_size();

	if( m_f01 == NULL )
	{
		m_f01 = engine()->create_buffer( 2*sizeof(nbcoord_t)*ps );
		m_predictor = engine()->create_buffer( sizeof(nbcoord_t)*ps );
		m_coeff = engine()->create_buffer( sizeof(nbcoord_t)*2 );
	}

	engine()->fcompute( t, y, m_f01, 0, 0 );
	engine()->fmadd( m_predictor, y, m_f01, dt, 0, 0, 0 );

	nbcoord_t	coeff[] = { dt*0.5, dt*0.5 };
	engine()->write_buffer( m_coeff, coeff );

	for( size_t s = 0; s <= m_refine_steps_count; ++s )
	{
		engine()->fcompute( t, m_predictor, m_f01, 0, ps );

		if( s == m_refine_steps_count )
		{
			engine()->fmaddn_inplace( y, m_f01, m_coeff, ps, 0, 0, 2 );
		}
		else
		{
			engine()->fmaddn( m_predictor, y, m_f01, m_coeff, ps, 0, 0, 0, 2 );
		}
	}
	engine()->advise_time( dt );
}
