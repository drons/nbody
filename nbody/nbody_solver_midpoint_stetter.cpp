#include "nbody_solver_midpoint_stetter.h"

nbody_solver_midpoint_stetter::nbody_solver_midpoint_stetter() :
	m_tmp(nullptr),
	m_du(nullptr),
	m_uv_initiated(false)
{
}

nbody_solver_midpoint_stetter::~nbody_solver_midpoint_stetter()
{
	engine()->free_buffers(m_fu);
	engine()->free_buffers(m_uv);
	engine()->free_buffer(m_tmp);
	engine()->free_buffer(m_du);
}

const char* nbody_solver_midpoint_stetter::type_name() const
{
	return "nbody_solver_midpoint_stetter";
}

void nbody_solver_midpoint_stetter::advise(nbcoord_t dt)
{
	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();

	bool	first_run(m_fu.empty());
	if(first_run)
	{
		size_t size = sizeof(nbcoord_t) * engine()->problem_size();
		m_tmp = engine()->create_buffer(size);
		m_du = engine()->create_buffer(size);
		m_fu = engine()->create_buffers(size, 2);
		m_uv = engine()->create_buffers(size, 2);
	}

	if(!m_uv_initiated || first_run)
	{
		engine()->copy_buffer(m_uv[0], y);
		engine()->copy_buffer(m_uv[1], y);
		first_run = true;
		m_uv_initiated = true;
	}

	//Compute u_{k+1}
	{
		if(first_run)
		{
			engine()->fcompute(t, m_uv[0], m_fu[0]);		// m_fu[0] = f(t, u_k)
		}
		engine()->fmadd(m_tmp, m_uv[1], m_fu[0], dt / 2_f);	// m_tmp = v_k + (dt/2) * m_fu[0]
		engine()->fcompute(t + dt / 2_f, m_tmp, m_du);		// m_du = f(t + dt/2, m_tmp)
		engine()->fmadd_inplace(m_uv[0], m_du, dt);			// u_{k+1} = u_k + dt * m_du
	}
	//Compute v[k+1]
	{
		nbcoord_t	v_coeff[] = {dt / 2_f, dt / 2_f};
		engine()->fcompute(t + dt, m_uv[0], m_fu[1]);		// m_fu[1] =  f(t, u_{k+1})
		engine()->fmaddn_inplace(m_uv[1], m_fu, v_coeff);	// v_{k+1} = v_k + (m_fu[1] + m_fu[0])/2
	}
	//Compute y
	nbcoord_t	y_coeff[] = {1_f / 2_f, 1_f / 2_f};
	engine()->fmaddn(y, nullptr, m_uv, y_coeff, m_uv.size());// y_{k+1} = (v_{k+1} + v_{k+1})/2

	//Use f(t, u_{k+1}) as f(t, u_k) at next step
	std::swap(m_fu[1], m_fu[0]);


	engine()->advise_time(dt);
}

void nbody_solver_midpoint_stetter::reset()
{
	m_uv_initiated = false;
}
