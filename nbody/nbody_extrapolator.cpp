#include "nbody_extrapolator.h"

nbody_extrapolator::nbody_extrapolator(
	nbody_engine* e, size_t order,
	const std::vector<size_t>& substeps_count):
	m_e(e),
	m_sub_steps_count(substeps_count)
{
	if(order != 1)
	{
		for(size_t& s : m_sub_steps_count)
		{
			s = std::pow(s, order);
		}
	}
}

nbody_extrapolator::~nbody_extrapolator()
{
}

nbody_engine* nbody_extrapolator::engine() const
{
	return m_e;
}

size_t nbody_extrapolator::sub_steps_count(size_t level) const
{
	return m_sub_steps_count[level];
}

nbody_extrapolator_berrut::nbody_extrapolator_berrut(
	nbody_engine* e, size_t order, const std::vector<size_t>& substeps_count):
	nbody_extrapolator(e, order, substeps_count)
{
	size_t size = sizeof(nbcoord_t) * e->problem_size();
	m_table = e->create_buffers(size, substeps_count.size());
	m_diff = e->create_buffer(size);
}

nbody_extrapolator_berrut::~nbody_extrapolator_berrut()
{
	engine()->free_buffers(m_table);
	engine()->free_buffer(m_diff);
}

void nbody_extrapolator_berrut::update_table(size_t level, nbody_engine::memory* y)
{
	engine()->copy_buffer(m_table[level], y);
}

nbcoord_t nbody_extrapolator_berrut::estimate_error(size_t level) const
{
	if(level == 0)
	{
		return std::numeric_limits<nbcoord_t>::max();
	}

	//compute maxabs( extrapolate(level) - extrapolate(level - 1) )

	nbcoord_t						maxdiff = 0_f;
	const std::vector<nbcoord_t>	w0(weights(level - 1));
	std::vector<nbcoord_t>			w1(weights(level));

	for(size_t n = 0; n != w0.size() ; ++n)
	{
		w1[n] -= w0[n];
	}
	nbody_engine::memory_array	table;
	for(size_t n = 0; n != level; ++n)
	{
		table.push_back(m_table[n]);
	}
	engine()->fill_buffer(m_diff, 0_f);
	engine()->fmaddn_inplace(m_diff, table, w1.data());
	engine()->fmaxabs(m_diff, maxdiff);
	return maxdiff;
}

void nbody_extrapolator_berrut::extrapolate(size_t level, nbody_engine::memory* ext_y) const
{
	const std::vector<nbcoord_t>	w(weights(level));
	nbody_engine::memory_array		table;
	for(size_t n = 0; n != level; ++n)
	{
		table.push_back(m_table[n]);
	}
	engine()->fill_buffer(ext_y, 0_f);
	engine()->fmaddn_inplace(ext_y, table, w.data());
}

std::vector<nbcoord_t> nbody_extrapolator_berrut::weights(size_t level) const
{
	std::vector<nbcoord_t>	w;
	w.resize(level);
	for(size_t n = 0; n != level; ++n)
	{
		w[n] = std::pow(-1, n + 1) * sub_steps_count(n);
	}
	auto sum = std::accumulate(w.begin(), w.end(), 0_f);
	for(size_t n = 0; n != level; ++n)
	{
		w[n] /= sum;
	}
	return w;
}

nbody_extrapolator_neville::nbody_extrapolator_neville(
	nbody_engine* e, size_t order, const std::vector<size_t>& substeps_count):
	nbody_extrapolator(e, order, substeps_count)
{
	size_t size = sizeof(nbcoord_t) * e->problem_size();
	m_table = e->create_buffers(size, substeps_count.size());
	m_table_prev = e->create_buffers(size, substeps_count.size());
	m_diff = e->create_buffer(size);
}

nbody_extrapolator_neville::~nbody_extrapolator_neville()
{
	engine()->free_buffers(m_table);
	engine()->free_buffers(m_table_prev);
	engine()->free_buffer(m_diff);
}

void nbody_extrapolator_neville::update_table(size_t k, nbody_engine::memory* y)
{
	std::swap(m_table, m_table_prev);
	engine()->copy_buffer(m_table[0], y);
	if(k == 0)
	{
		return;
	}
	/*
	   From 'Numerical Recipes'

		(17.3.8)

		                      T_{k,j} - T_{k-1,j}
		T_{k,j+1} = T_{k,j} + -------------------
		                      (n_k/n_{k-j})^2 -1

		            T_{k,j}(n_k^2 - n_{k-j}^2) + T_{k,j}n_{k-j}^2 - T_{k-1,j}n_{k-j}^2
		T_{k,j+1} = ------------------------------------------------------------------
		                                    n_k^2 - n_{k-j}^2

		            T_{k,j}n_k^2 - T_{k-1,j}n_{k-j}^2
		T_{k,j+1} = ---------------------------------
		                   n_k^2 - n_{k-j}^2

	   (17.3.7)

	   T_00
	   T_10 T_11
	   T_20 T_21 T_22
	   ...  ...  ...  ...
	   T_k0= y_k
	*/
	for(size_t j = 0; j != k; ++j)
	{
		nbcoord_t	step1(static_cast<nbcoord_t>(sub_steps_count(k)));
		nbcoord_t	step2(static_cast<nbcoord_t>(sub_steps_count(k - j - 1)));
		nbcoord_t	step_diff = step1 - step2;
		nbcoord_t	d[2] = {step1 / step_diff, -step2 / step_diff};
		nbody_engine::memory_array c = {m_table[j], m_table_prev[j]};
		engine()->fmaddn(m_table[j + 1], nullptr, c, d, 2);
	}
}

nbcoord_t nbody_extrapolator_neville::estimate_error(size_t level) const
{
	if(level == 0)
	{
		return std::numeric_limits<nbcoord_t>::max();
	}
	nbcoord_t	maxdiff = 0_f;
	engine()->fmadd(m_diff, m_table[level], m_table[level - 1], -1_f);
	engine()->fmaxabs(m_diff, maxdiff);
	return maxdiff;
}

void nbody_extrapolator_neville::extrapolate(size_t level, nbody_engine::memory* ext_y) const
{
	engine()->copy_buffer(ext_y, m_table[level]);
}

nbody_extrapolator* nbody_create_extrapolator(
	const QString& type, nbody_engine* engine, size_t order,
	const std::vector<size_t>& substeps_count)
{
	if(type == "berrut")
	{
		return new nbody_extrapolator_berrut(engine, order, substeps_count);
	}
	else if(type == "neville")
	{
		return new nbody_extrapolator_neville(engine, order, substeps_count);
	}
	return nullptr;
}
