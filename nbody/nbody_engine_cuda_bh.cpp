#include "nbody_engine_cuda_bh.h"

#include <QDebug>

#include "nbody_engine_cuda_impl.h"
#include "nbody_space_heap.h"

nbody_engine_cuda_bh::nbody_engine_cuda_bh() :
	m_cycle_traverse(false),
	m_distance_to_node_radius_ratio(10),
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

	advise_compute_count();

	size_t					count = m_data->get_count();
	std::vector<nbcoord_t>	host_y(y->size() / sizeof(nbcoord_t));
	std::vector<nbcoord_t>	host_mass(count);

	read_buffer(host_y.data(), y);
	read_buffer(host_mass.data(), m_mass);

	const nbcoord_t*	rx = host_y.data();
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	mass = host_mass.data();

	nbody_space_heap	heap;
	heap.build(count, rx, ry, rz, mass, m_distance_to_node_radius_ratio);

	size_t			tree_size = heap.get_radius_sqr().size();

	if(m_dev_indites == NULL)
	{
		m_dev_tree_cmx = create_buffer(tree_size * sizeof(nbcoord_t));
		m_dev_tree_cmy = create_buffer(tree_size * sizeof(nbcoord_t));
		m_dev_tree_cmz = create_buffer(tree_size * sizeof(nbcoord_t));
		m_dev_tree_crit_r2 = create_buffer(tree_size * sizeof(nbcoord_t));
		m_dev_tree_mass = create_buffer(tree_size * sizeof(nbcoord_t));
		m_dev_indites = create_buffer(tree_size * sizeof(int));
	}

	nbcoord_t*		dev_y = static_cast<nbcoord_t*>(y->data());
	nbcoord_t*		dev_f = static_cast<nbcoord_t*>(f->data());
	nbcoord_t*		dev_tree_cmx = static_cast<nbcoord_t*>(m_dev_tree_cmx->data());
	nbcoord_t*		dev_tree_cmy = static_cast<nbcoord_t*>(m_dev_tree_cmy->data());
	nbcoord_t*		dev_tree_cmz = static_cast<nbcoord_t*>(m_dev_tree_cmz->data());
	nbcoord_t*		dev_tree_mass = static_cast<nbcoord_t*>(m_dev_tree_mass->data());
	nbcoord_t*		dev_tree_crit_r2 = static_cast<nbcoord_t*>(m_dev_tree_crit_r2->data());
	int*			dev_indites = static_cast<int*>(m_dev_indites->data());

	std::vector<nbcoord_t>	host_tree_cmx(tree_size), host_tree_cmy(tree_size), host_tree_cmz(tree_size);
	std::vector<int>		host_indites(tree_size);

	for(size_t n = 0; n != tree_size; ++n)
	{
		host_tree_cmx[n] = heap.get_mass_center()[n].x;
		host_tree_cmy[n] = heap.get_mass_center()[n].y;
		host_tree_cmz[n] = heap.get_mass_center()[n].z;
		host_indites[n] = heap.get_body_n()[n];
	}

	write_buffer(m_dev_tree_cmx, host_tree_cmx.data());
	write_buffer(m_dev_tree_cmy, host_tree_cmy.data());
	write_buffer(m_dev_tree_cmz, host_tree_cmz.data());
	write_buffer(m_dev_tree_mass, heap.get_mass().data());
	write_buffer(m_dev_tree_crit_r2, heap.get_radius_sqr().data());
	write_buffer(m_dev_indites, host_indites.data());

	fcompute_heap_bh(0, count, tree_size, dev_y, dev_f,
					 dev_tree_cmx, dev_tree_cmy, dev_tree_cmz,
					 dev_tree_mass, dev_tree_crit_r2, dev_indites);
}

void nbody_engine_cuda_bh::print_info() const
{
	nbody_engine_cuda::print_info();
	qDebug() << "\t" << "distance_to_node_radius_ratio:" << m_distance_to_node_radius_ratio;
	qDebug() << "\t" << "traverse_type:" << ((m_cycle_traverse) ? "cycle" : "nested_tree");
	qDebug() << "\t" << "tree_layout:" << "heap";
}
