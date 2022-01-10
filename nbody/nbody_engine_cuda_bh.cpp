#include "nbody_engine_cuda_bh.h"

#include <omp.h>
#include <QDebug>

#include "nbody_engine_cuda_memory.h"
#include "nbody_space_heap.h"

nbody_engine_cuda_bh::nbody_engine_cuda_bh(nbcoord_t distance_to_node_radius_ratio) :
	m_cycle_traverse(false),
	m_distance_to_node_radius_ratio(distance_to_node_radius_ratio),
	m_dev_tree_cmx(NULL),
	m_dev_tree_cmy(NULL),
	m_dev_tree_cmz(NULL),
	m_dev_tree_mass(NULL),
	m_dev_tree_crit_r2(NULL),
	m_dev_indites(NULL)
{
}

nbody_engine_cuda_bh::~nbody_engine_cuda_bh()
{
	delete m_dev_tree_cmx;
	delete m_dev_tree_cmy;
	delete m_dev_tree_cmz;
	delete m_dev_tree_mass;
	delete m_dev_tree_crit_r2;
	delete m_dev_indites;
}

const char* nbody_engine_cuda_bh::type_name() const
{
	return "nbody_engine_cuda_bh";
}

void nbody_engine_cuda_bh::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const smemory*>(_y);
	smemory*		f = dynamic_cast<smemory*>(_f);

	if(y == NULL)
	{
		qDebug() << "y is not smemory";
		return;
	}

	if(f == NULL)
	{
		qDebug() << "f is not smemory";
		return;
	}

	size_t	device_count(m_device_ids.size());

	if(device_count > 1)
	{
		// synchronize multiple devices
		synchronize_y(const_cast<smemory*>(y));
	}

	advise_compute_count();

	size_t					data_size = m_data->get_count();
	std::vector<nbcoord_t>	host_y(y->size() / sizeof(nbcoord_t));
	std::vector<nbcoord_t>	host_mass(data_size);

	read_buffer(host_y.data(), y);
	read_buffer(host_mass.data(), m_mass);

	const nbcoord_t*	rx = host_y.data();
	const nbcoord_t*	ry = rx + data_size;
	const nbcoord_t*	rz = rx + 2 * data_size;
	const nbcoord_t*	mass = host_mass.data();

	nbody_space_heap	heap;
	heap.build(data_size, rx, ry, rz, mass, m_distance_to_node_radius_ratio);

	size_t			tree_size = heap.get_radius_sqr().size();

	if(m_dev_indites == NULL)
	{
		m_dev_tree_cmx = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t)));
		m_dev_tree_cmy = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t)));
		m_dev_tree_cmz = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t)));
		m_dev_tree_crit_r2 = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t)));
		m_dev_tree_mass = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t)));
		m_dev_indites = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(int)));
	}

	std::vector<nbcoord_t>	host_tree_cmx(tree_size), host_tree_cmy(tree_size), host_tree_cmz(tree_size);
	std::vector<int>		host_indites(tree_size);

	#pragma omp parallel for
	for(size_t n = 0; n < tree_size; ++n)
	{
		host_tree_cmx[n] = heap.get_mass_center()[n].x;
		host_tree_cmy[n] = heap.get_mass_center()[n].y;
		host_tree_cmz[n] = heap.get_mass_center()[n].z;
		host_indites[n] = static_cast<int>(heap.get_body_n()[n]);
	}

	write_buffer(m_dev_tree_cmx, host_tree_cmx.data());
	write_buffer(m_dev_tree_cmy, host_tree_cmy.data());
	write_buffer(m_dev_tree_cmz, host_tree_cmz.data());
	write_buffer(m_dev_tree_mass, heap.get_mass().data());
	write_buffer(m_dev_tree_crit_r2, heap.get_radius_sqr().data());
	write_buffer(m_dev_indites, host_indites.data());

	fill_buffer(f, 0);
	size_t	device_data_size = data_size / device_count;

	#pragma omp parallel num_threads(device_count)
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	offset = dev_n * device_data_size;
		cudaSetDevice(m_device_ids[dev_n]);
		const nbcoord_t*	dev_y = static_cast<const nbcoord_t*>(y->data(dev_n));
		nbcoord_t*			dev_f = static_cast<nbcoord_t*>(f->data(dev_n));
		nbcoord_t*			dev_tree_cmx = static_cast<nbcoord_t*>(m_dev_tree_cmx->data(dev_n));
		nbcoord_t*			dev_tree_cmy = static_cast<nbcoord_t*>(m_dev_tree_cmy->data(dev_n));
		nbcoord_t*			dev_tree_cmz = static_cast<nbcoord_t*>(m_dev_tree_cmz->data(dev_n));
		nbcoord_t*			dev_tree_mass = static_cast<nbcoord_t*>(m_dev_tree_mass->data(dev_n));
		nbcoord_t*			dev_tree_crit_r2 = static_cast<nbcoord_t*>(m_dev_tree_crit_r2->data(dev_n));
		int*				dev_indites = static_cast<int*>(m_dev_indites->data(dev_n));

		fcompute_heap_bh(static_cast<int>(offset), static_cast<int>(data_size),
						 static_cast<int>(device_data_size), static_cast<int>(tree_size),
						 dev_y, dev_f,
						 dev_tree_cmx, dev_tree_cmy, dev_tree_cmz,
						 dev_tree_mass, dev_tree_crit_r2,
						 dev_indites, get_block_size());
		cudaDeviceSynchronize();
	}
	if(device_count > 1)
	{
		// synchronize again
		synchronize_sum(f);
	}
}

void nbody_engine_cuda_bh::print_info() const
{
	nbody_engine_cuda::print_info();
	qDebug() << "\t" << "distance_to_node_radius_ratio:" << m_distance_to_node_radius_ratio;
	qDebug() << "\t" << "traverse_type:" << ((m_cycle_traverse) ? "cycle" : "nested_tree");
	qDebug() << "\t" << "tree_layout:" << "heap";
}
