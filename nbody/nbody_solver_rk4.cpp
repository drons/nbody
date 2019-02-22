#include "nbody_solver_rk4.h"

nbody_solver_rk4::nbody_solver_rk4() : nbody_solver()
{
	m_tmp = NULL;
}

nbody_solver_rk4::~nbody_solver_rk4()
{
	engine()->free_buffers(m_k);
	engine()->free_buffer(m_tmp);
}

const char* nbody_solver_rk4::type_name() const
{
	return "nbody_solver_rk4";
}

void nbody_solver_rk4::advise(nbcoord_t dt)
{
	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();
	size_t					ps = engine()->problem_size();

	if(m_k.empty())
	{
		m_k = engine()->create_buffers(sizeof(nbcoord_t) * ps, 4);
		m_tmp = engine()->create_buffer(sizeof(nbcoord_t) * ps);
	}

	engine()->fcompute(t, y, m_k[0]);   // k1 = f( t, y )

	engine()->fmadd(m_tmp, y, m_k[0], dt / 2_f); //tmp = y + 0.5*k1*dt
	engine()->fcompute(t + dt / 2_f, m_tmp, m_k[1]); // k2 = f( t + 0.5*dt, y + 0.5*k1*dt )

	engine()->fmadd(m_tmp, y, m_k[1], dt / 2_f); //tmp = y + 0.5*k2*dt
	engine()->fcompute(t + dt / 2_f, m_tmp, m_k[2]); // k3 = f( t + 0.5*dt, y + 0.5*k2*dt )

	engine()->fmadd(m_tmp, y, m_k[2], dt); //tmp = y + k3*dt
	engine()->fcompute(t + dt, m_tmp, m_k[3]); // k4 = f( t + dt, y + k3*dt )

	//y += 	dt( k1/6 + k2/3 + k3/3 + k4/8 )
	const nbcoord_t	coeff[] = { dt / 6_f, dt / 3_f, dt / 3_f, dt / 6_f };

	engine()->fmaddn_inplace(y, m_k, coeff);

	engine()->advise_time(dt);
}

