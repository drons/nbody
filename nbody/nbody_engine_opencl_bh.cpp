#include "nbody_engine_opencl_bh.h"

nbody_engine_opencl_bh::nbody_engine_opencl_bh(nbcoord_t distance_to_node_radius_ratio) :
	m_distance_to_node_radius_ratio(distance_to_node_radius_ratio),
	m_cycle_traverse(true)
{
}

nbody_engine_opencl_bh::~nbody_engine_opencl_bh()
{
}

const char* nbody_engine_opencl_bh::type_name() const
{
	return "nbody_engine_opencl_bh";
}

void nbody_engine_opencl_bh::fcompute(const nbcoord_t& t, const memory* y, memory* f)
{
	fcompute_bh_impl(t, y, f, m_distance_to_node_radius_ratio, m_cycle_traverse);
}

void nbody_engine_opencl_bh::print_info() const
{
	nbody_engine_opencl::print_info();
	qDebug() << "\tdistance_to_node_radius_ratio:" << m_distance_to_node_radius_ratio;
	qDebug() << "\ttraverse_type:" << ((m_cycle_traverse) ? "cycle" : "nested_tree");
	qDebug() << "\ttree_layout:" << "heap";
}

void nbody_engine_opencl_bh::set_cycle_traverse(bool tt)
{
	m_cycle_traverse = tt;
}
