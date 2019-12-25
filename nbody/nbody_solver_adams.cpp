#include "nbody_solver_adams.h"
#include "nbody_solver_euler.h"
#include "summation.h"
#include <QDebug>

nbody_solver_adams::nbody_solver_adams(nbody_solver* starter,
									   size_t rank,
									   bool corr) :
	nbody_solver(),
	m_starter(starter),
	m_corr_data(nullptr),
	m_rank(rank),
	m_correction(corr)
{
}

nbody_solver_adams::~nbody_solver_adams()
{
	delete m_starter;
	engine()->free_buffers(m_f);
	engine()->free_buffer(m_corr_data);
}

const char* nbody_solver_adams::type_name() const
{
	return "nbody_solver_adams";
}

void nbody_solver_adams::advise(nbcoord_t dt)
{
	const nbcoord_t		a1[1] = { 1_f };
	const nbcoord_t		a2[2] = { 3_f / 2_f, -1_f / 2_f };
	const nbcoord_t		a3[3] = { 23_f / 12_f, -4_f / 3_f, 5_f / 12_f };
	const nbcoord_t		a4[4] = { 55_f / 24_f, -59_f / 24_f, 37_f / 24_f, -3_f / 8_f };
	const nbcoord_t		a5[5] = { 1901_f / 720_f, -1387_f / 360_f, 109_f / 30_f, -637_f / 360_f, 251_f / 720_f };
	const nbcoord_t*	ar[] = { NULL, a1, a2, a3, a4, a5 };
	const nbcoord_t*	a = ar[m_rank];

	nbody_engine::memory*	y = engine()->get_y();
	nbcoord_t				t = engine()->get_time();
	size_t					step = engine()->get_step();
	size_t					fnum = step % m_rank;
	size_t					ps = engine()->problem_size();

	if(m_f.empty())
	{
		m_starter->set_engine(engine());
		m_f = engine()->create_buffers(sizeof(nbcoord_t) * ps, m_rank);
		if(m_correction)
		{
			m_corr_data = engine()->create_buffer(sizeof(nbcoord_t) * ps);
			engine()->fill_buffer(m_corr_data, 0);
		}
	}

	if(step > m_rank)
	{
		std::vector<nbcoord_t>	coeff(m_rank);

		engine()->fcompute(t, y, m_f[fnum]);

		for(size_t n = 0; n < m_rank; ++n)
		{
			coeff[(m_rank + fnum - n) % m_rank ] = a[n] * dt;
		}

		if(m_correction)
		{
			engine()->fmaddn_corr(y, m_corr_data, m_f, coeff.data());
		}
		else
		{
			engine()->fmaddn_inplace(y, m_f, coeff.data());
		}

		engine()->advise_time(dt);
	}
	else
	{
		engine()->fcompute(t, y, m_f[fnum]);
		m_starter->advise(dt);
	}
}

void nbody_solver_adams::print_info() const
{
	nbody_solver::print_info();
	qDebug() << "\trank" << m_rank;
	qDebug() << "\tcorrection" << m_correction;
	qDebug() << "\tstarter" << m_starter->type_name();
}
