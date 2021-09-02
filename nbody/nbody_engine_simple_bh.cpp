#include "nbody_engine_simple_bh.h"

#include <QDebug>

#include "nbody_space_heap.h"
#include "nbody_space_heap_stackless.h"
#include "nbody_space_tree.h"

nbody_engine_simple_bh::nbody_engine_simple_bh(nbcoord_t distance_to_node_radius_ratio,
											   e_traverse_type tt,
											   e_tree_layout tl) :
	m_distance_to_node_radius_ratio(distance_to_node_radius_ratio),
	m_traverse_type(tt),
	m_tree_layout(tl)
{
}

template<class T>
void nbody_engine_simple_bh::space_subdivided_fcompute(const smemory* y, smemory* f)
{
	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = reinterpret_cast<const nbcoord_t*>(y->data());
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	vx = rx + 3 * count;
	const nbcoord_t*	vy = rx + 4 * count;
	const nbcoord_t*	vz = rx + 5 * count;

	nbcoord_t*			frx = reinterpret_cast<nbcoord_t*>(f->data());
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2 * count;
	nbcoord_t*			fvx = frx + 3 * count;
	nbcoord_t*			fvy = frx + 4 * count;
	nbcoord_t*			fvz = frx + 5 * count;

	const nbcoord_t*	mass = reinterpret_cast<const nbcoord_t*>(m_mass->data());
	T					tree;

	tree.build(count, rx, ry, rz, mass, m_distance_to_node_radius_ratio);

	auto update_f = [ = ](size_t body1, const nbvertex_t& total_force, nbcoord_t mass1)
	{
		frx[body1] = vx[body1];
		fry[body1] = vy[body1];
		frz[body1] = vz[body1];
		fvx[body1] = total_force.x / mass1;
		fvy[body1] = total_force.y / mass1;
		fvz[body1] = total_force.z / mass1;
	};

	auto node_visitor = [&](size_t body1, const nbvertex_t& v1, const nbcoord_t mass1)
	{
		nbvertex_t			total_force(tree.traverse(m_data, v1, mass1));
		update_f(body1, total_force, mass[body1]);
	};

	if(ett_cycle == m_traverse_type)
	{
		#pragma omp parallel for schedule(dynamic, 4)
		for(size_t body1 = 0; body1 < count; ++body1)
		{
			const nbvertex_t	v1(rx[body1], ry[body1], rz[body1]);
			const nbcoord_t		mass1(mass[body1]);
			const nbvertex_t	total_force(tree.traverse(m_data, v1, mass1));
			update_f(body1, total_force, mass1);
		}
	}
	else if(ett_nested_tree == m_traverse_type)
	{
		tree.traverse(node_visitor);
	}
}

const char* tree_layout_name(e_tree_layout tree_layout)
{
	switch(tree_layout)
	{
	case etl_tree:
		return "tree";
	case etl_heap:
		return "heap";
	case etl_heap_stackless:
		return "heap_stackless";
	default:
		return "";
	}
	return "";
}

e_tree_layout tree_layout_from_str(const QString& name)
{
	if(name == "tree")
	{
		return etl_tree;
	}
	else if(name == "heap")
	{
		return etl_heap;
	}
	else if(name == "heap_stackless")
	{
		return etl_heap_stackless;
	}

	return etl_unknown;
}

void nbody_engine_simple_bh::print_info() const
{
	nbody_engine_openmp::print_info();
	qDebug() << "\tdistance_to_node_radius_ratio:" << m_distance_to_node_radius_ratio;
	qDebug() << "\ttraverse_type:" << (m_traverse_type == ett_cycle ? "cycle" : "nested_tree");
	qDebug() << "\ttree_layout:" << tree_layout_name(m_tree_layout);
}


nbody_engine_simple_bh_tree::nbody_engine_simple_bh_tree(nbcoord_t ratio,
														 e_traverse_type tt) :
	nbody_engine_simple_bh(ratio, tt, etl_tree)
{
}

const char* nbody_engine_simple_bh_tree::type_name() const
{
	return "nbody_engine_simple_bh_tree";
}

void nbody_engine_simple_bh_tree::fcompute(const nbcoord_t& t,
										   const nbody_engine::memory* _y,
										   nbody_engine::memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
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

	space_subdivided_fcompute<nbody_space_tree>(y, f);
}

nbody_engine_simple_bh_heap::nbody_engine_simple_bh_heap(nbcoord_t ratio,
														 e_traverse_type tt) :
	nbody_engine_simple_bh(ratio, tt, etl_heap)
{
}

const char* nbody_engine_simple_bh_heap::type_name() const
{
	return "nbody_engine_simple_bh_heap";
}

void nbody_engine_simple_bh_heap::fcompute(const nbcoord_t& t,
										   const nbody_engine::memory* _y,
										   nbody_engine::memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
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

	space_subdivided_fcompute<nbody_space_heap>(y, f);
}

nbody_engine_simple_bh_heap_stackless::nbody_engine_simple_bh_heap_stackless(nbcoord_t ratio,
																			 e_traverse_type tt) :
	nbody_engine_simple_bh(ratio, tt, etl_heap_stackless)
{
}

const char* nbody_engine_simple_bh_heap_stackless::type_name() const
{
	return "nbody_engine_simple_bh_heap_stackless";
}

void nbody_engine_simple_bh_heap_stackless::fcompute(const nbcoord_t& t,
													 const nbody_engine::memory* _y,
													 nbody_engine::memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
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

	space_subdivided_fcompute<nbody_space_heap_stackless>(y, f);
}
