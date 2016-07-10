#include "nbody_solver_rk4.h"

nbody_solver_rk4::nbody_solver_rk4() : nbody_solver()
{
	m_k = NULL;
	m_tmp = NULL;
	m_coeff = NULL;
}

nbody_solver_rk4::~nbody_solver_rk4()
{
	engine()->free( m_k );
	engine()->free( m_tmp );
	engine()->free( m_coeff );
}

const char* nbody_solver_rk4::type_name() const
{
	return "nbody_solver_rk4";
}

void nbody_solver_rk4::step( double dt )
{
	nbody_engine::memory*	y = engine()->y();
	nbcoord_t				t = 0;
	size_t					ps = engine()->problem_size();

	if( m_k == NULL )
	{
		m_k = engine()->malloc( 4*sizeof(nbcoord_t)*ps );
		m_tmp = engine()->malloc( sizeof(nbcoord_t)*ps );
		m_coeff = engine()->malloc( sizeof(nbcoord_t)*4 );
	}

	engine()->fcompute( t, y, m_k, 0, 0 ); // k1 = f( t, y )

	engine()->fmadd( m_tmp, y, m_k, 0.5*dt, 0, 0, 0 ); //tmp = y + 0.5*k1*dt
	engine()->fcompute( t + 0.5*dt, m_tmp, m_k, 0, ps ); // k2 = f( t + 0.5*dt, y + 0.5*k1*dt )

	engine()->fmadd( m_tmp, y, m_k, 0.5*dt, 0, 0, ps ); //tmp = y + 0.5*k2*dt
	engine()->fcompute( t + 0.5*dt, m_tmp, m_k, 0, 2*ps ); // k3 = f( t + 0.5*dt, y + 0.5*k2*dt )

	engine()->fmadd( m_tmp, y, m_k, dt, 0, 0, 2*ps ); //tmp = y + k3*dt
	engine()->fcompute( t + dt, m_tmp, m_k, 0, 3*ps ); // k4 = f( t + dt, y + k3*dt )

	//y += 	dt( k1/6 + k2/3 + k3/3 + k4/8 )
	nbcoord_t	coeff[] = { dt/6.0, dt/3.0, dt/3.0, dt/6.0 };

	engine()->memcpy( m_coeff, coeff );
	engine()->fmaddn( y, m_k, m_coeff, ps, 0, 0, 4 );

	//engine()->fmadd( y, m_k, dt ); //y += dy*dt

	engine()->advise_time( dt );
}

