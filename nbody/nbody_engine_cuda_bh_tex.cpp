#include "nbody_engine_cuda_bh_tex.h"

#include <QDebug>

#include "nbody_engine_cuda_memory.h"
#include "nbody_space_heap.h"

nbody_engine_cuda_bh_tex::nbody_engine_cuda_bh_tex(nbcoord_t distance_to_node_radius_ratio, e_tree_layout tl) :
	m_cycle_traverse(false),
	m_tree_layout(tl),
	m_distance_to_node_radius_ratio(distance_to_node_radius_ratio),
	m_dev_tree_xyzr(NULL),
	m_dev_tree_mass(NULL),
	m_dev_indites(NULL)
{
}

nbody_engine_cuda_bh_tex::~nbody_engine_cuda_bh_tex()
{
	delete m_dev_tree_xyzr;
	delete m_dev_tree_mass;
	delete m_dev_indites;
}

const char* nbody_engine_cuda_bh_tex::type_name() const
{
	return "nbody_engine_cuda_bh_tex";
}

void nbody_engine_cuda_bh_tex::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
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
		m_dev_tree_xyzr = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t) * 4));
		m_dev_tree_mass = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(nbcoord_t)));
		m_dev_indites = dynamic_cast<smemory*>(create_buffer(tree_size * sizeof(int)));
	}

	const nbcoord_t*	dev_y = static_cast<const nbcoord_t*>(y->data());
	nbcoord_t*			dev_f = static_cast<nbcoord_t*>(f->data());
	int*				dev_indites = static_cast<int*>(m_dev_indites->data());

	static_assert(sizeof(vertex4<nbcoord_t>) == sizeof(nbcoord_t) * 4,
				  "sizeof(vertex4) must be equal to sizeof(nbcoord_t)*4");

	std::vector<vertex4<nbcoord_t>>	host_tree_xyzr(tree_size);
	std::vector<int>				host_indites(tree_size);

	for(size_t n = 0; n != tree_size; ++n)
	{
		host_tree_xyzr[n].x = heap.get_mass_center()[n].x;
		host_tree_xyzr[n].y = heap.get_mass_center()[n].y;
		host_tree_xyzr[n].z = heap.get_mass_center()[n].z;
		host_tree_xyzr[n].w = heap.get_radius_sqr()[n];
		host_indites[n] = static_cast<int>(heap.get_body_n()[n]);
	}

	write_buffer(m_dev_tree_xyzr, host_tree_xyzr.data());
	write_buffer(m_dev_tree_mass, heap.get_mass().data());
	write_buffer(m_dev_indites, host_indites.data());

	if(m_tree_layout == etl_heap)
	{
		fcompute_heap_bh_tex(0, static_cast<int>(count), static_cast<int>(tree_size), dev_f,
							 m_dev_tree_xyzr->tex(4), m_dev_tree_mass->tex(),
							 dev_indites, get_block_size());
	}
	else if(m_tree_layout == etl_heap_stackless)
	{
		fcompute_heap_bh_stackless(0, static_cast<int>(count), static_cast<int>(tree_size), dev_f,
								   m_dev_tree_xyzr->tex(4), m_dev_tree_mass->tex(),
								   dev_indites, get_block_size());
	}

	fcompute_xyz(dev_y, dev_f, static_cast<int>(count), get_block_size());
}

void nbody_engine_cuda_bh_tex::print_info() const
{
	nbody_engine_cuda::print_info();
	qDebug() << "\t" << "distance_to_node_radius_ratio:" << m_distance_to_node_radius_ratio;
	qDebug() << "\t" << "traverse_type:" << ((m_cycle_traverse) ? "cycle" : "nested_tree");
	qDebug() << "\t" << "tree_layout:" << tree_layout_name(m_tree_layout);
}
