#include "nbody_engine_simple_bh.h"

#include <memory>
#include <QDebug>

#include "summation.h"

static constexpr	size_t SPACE_DIMENSIONS = 3;
static constexpr	size_t MAX_STACK_SIZE = 64;

class nbody_space_tree
{
	class node
	{
		friend class			nbody_space_tree;
		size_t					m_dimension;
		node*					m_left;
		node*					m_right;
		nbvertex_t				m_mass_center;
		nbvertex_t				m_min;
		nbvertex_t				m_max;
		nbcoord_t				m_mass;
		nbcoord_t				m_radius_sqr;
		size_t					m_count;
		size_t					m_body_n;
	public:
		explicit node(size_t dim = 0) :
			m_dimension(dim),
			m_left(nullptr),
			m_right(nullptr),
			m_min(std::numeric_limits<nbcoord_t>::max(),
				  std::numeric_limits<nbcoord_t>::max(),
				  std::numeric_limits<nbcoord_t>::max()),
			m_max(-std::numeric_limits<nbcoord_t>::max(),
				  -std::numeric_limits<nbcoord_t>::max(),
				  -std::numeric_limits<nbcoord_t>::max()),
			m_mass(0),
			m_radius_sqr(0),
			m_count(0),
			m_body_n(std::numeric_limits<size_t>::max())
		{
		}
		~node()
		{
			delete m_left;
			delete m_right;
		}
		void build(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				   const nbcoord_t* mass);
	};
	node*	m_root;

public:
	nbody_space_tree() :
		m_root(nullptr)
	{
	}
	~nbody_space_tree()
	{
		delete m_root;
	}

	void build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz, const nbcoord_t* mass)
	{
		std::vector<size_t>	bodies_indites;

		bodies_indites.resize(count);
		for(size_t i = 0; i != count; ++i)
		{
			bodies_indites[i] = i;
		}

		m_root = new node(0);
		#pragma omp parallel
		#pragma omp single
		m_root->build(count, bodies_indites.data(), rx, ry, rz, mass);
	}

	template<class Visitor>
	void traverse(Visitor visit) const
	{
		node*	stack_data[MAX_STACK_SIZE] = {};
		node**	stack = stack_data;
		node**	stack_head = stack;

		*stack++ = m_root;
		while(stack != stack_head)
		{
			node*				curr = *--stack;
			if(curr->m_radius_sqr > 0)
			{
				if(curr->m_left != NULL)
				{
					*stack++ = curr->m_left;
				}
				if(curr->m_right != NULL)
				{
					*stack++ = curr->m_right;
				}
			}
			else
			{
				visit(curr->m_body_n, curr->m_mass_center, curr->m_mass);
			}
		}
	}

	nbvertex_t traverse(const nbody_data* data, nbcoord_t distance_to_node_radius_ratio,
						const nbvertex_t& v1, const nbcoord_t mass1) const
	{
		nbvertex_t			total_force;

		node*	stack_data[MAX_STACK_SIZE] = {};
		node**	stack = stack_data;
		node**	stack_head = stack;

		*stack++ = m_root;
		while(stack != stack_head)
		{
			node*				curr = *--stack;
			const nbcoord_t		distance_sqr((v1 - curr->m_mass_center).norm());

			if(distance_sqr > distance_to_node_radius_ratio * curr->m_radius_sqr)
			{
				total_force += data->force(v1, curr->m_mass_center, mass1, curr->m_mass);
			}
			else
			{
				if(curr->m_left != NULL)
				{
					*stack++ = curr->m_left;
				}
				if(curr->m_right != NULL)
				{
					*stack++ = curr->m_right;
				}
			}
		}
		return total_force;
	}
};

struct mass_center_mass_proxy
{
	const size_t* indites;
	const nbcoord_t* rx;
	const nbcoord_t* ry;
	const nbcoord_t* rz;
	const nbcoord_t* mass;
public:
	mass_center_mass_proxy(size_t* _indites,
						   const nbcoord_t* _rx, const nbcoord_t* _ry, const nbcoord_t* _rz,
						   const nbcoord_t* _mass):
		indites(_indites),
		rx(_rx),
		ry(_ry),
		rz(_rz),
		mass(_mass)
	{
	}
	vertex4<nbcoord_t> operator [](size_t n) const
	{
		size_t	i(indites[n]);
		return vertex4<nbcoord_t>(rx[i] * mass[i],
								  ry[i] * mass[i],
								  rz[i] * mass[i],
								  mass[i]);
	}
};

void nbody_space_tree::node::build(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry,
								   const nbcoord_t* rz, const nbcoord_t* mass)
{
	m_count = count;

	if(count == 1) // It is a leaf
	{
		m_mass_center = nbvertex_t(rx[*indites], ry[*indites], rz[*indites]);
		m_mass = mass[*indites];
		m_body_n = *indites;
		return;
	}

	vertex4<nbcoord_t>	mass_center_mass(summation<vertex4<nbcoord_t>, mass_center_mass_proxy>(
											 mass_center_mass_proxy(indites, rx, ry, rz, mass), count));

	m_mass_center.x = mass_center_mass.x / mass_center_mass.w;
	m_mass_center.y = mass_center_mass.y / mass_center_mass.w;
	m_mass_center.z = mass_center_mass.z / mass_center_mass.w;
	m_mass = mass_center_mass.w;

	nbcoord_t	max_rad_sqr = 0;
	for(size_t n = 0; n != count; ++n)
	{
		size_t				i(indites[n]);
		const nbvertex_t	v(rx[i], ry[i], rz[i]);

		nbcoord_t			rad_sqr((m_mass_center - v).norm());
		max_rad_sqr = std::max(max_rad_sqr, rad_sqr);

		for(size_t d = 0; d != SPACE_DIMENSIONS; ++d)
		{
			if(m_max[d] < v[d])
			{
				m_max[d] = v[d];
			}
			else if(m_min[d] > v[d])
			{
				m_min[d] = v[d];
			}
		}
	}

	m_radius_sqr = max_rad_sqr;

	std::vector<size_t>	left, right;
	left.reserve(count / 2);
	right.reserve(count / 2);

	for(size_t n = 0; n != count; ++n)
	{
		size_t				i(indites[n]);
		const nbvertex_t	v(rx[i], ry[i], rz[i]);
		if(v[m_dimension] < m_mass_center[m_dimension])
		{
			left.push_back(i);
		}
		else
		{
			right.push_back(i);
		}
	}

	size_t next_dimension((m_dimension + 1) % SPACE_DIMENSIONS);
	m_left = new node(next_dimension);
	m_right = new node(next_dimension);

	if(count > NBODY_DATA_BLOCK_SIZE)
	{
		#pragma omp task
		m_left->build(left.size(), left.data(), rx, ry, rz, mass);
		#pragma omp task
		m_right->build(right.size(), right.data(), rx, ry, rz, mass);
		#pragma omp taskwait
	}
	else
	{
		m_left->build(left.size(), left.data(), rx, ry, rz, mass);
		m_right->build(right.size(), right.data(), rx, ry, rz, mass);
	}
}


nbody_engine_simple_bh::nbody_engine_simple_bh(nbcoord_t distance_to_node_radius_ratio, e_traverse_type tt) :
	m_distance_to_node_radius_ratio(distance_to_node_radius_ratio), m_traverse_type(tt)
{
}

const char* nbody_engine_simple_bh::type_name() const
{
	return "nbody_engine_simple_bh";
}

void nbody_engine_simple_bh::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
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

	nbody_space_tree	tree;

	tree.build(count, rx, ry, rz, mass);

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
		nbvertex_t			total_force(tree.traverse(m_data, m_distance_to_node_radius_ratio, v1, mass1));
		update_f(body1, total_force, mass[body1]);
	};

	switch(m_traverse_type)
	{
	case ett_cycle:
		#pragma omp parallel for schedule(dynamic, 4)
		for(size_t body1 = 0; body1 < count; ++body1)
		{
			const nbvertex_t	v1(rx[body1], ry[body1], rz[body1]);
			const nbcoord_t		mass1(mass[body1]);
			const nbvertex_t	total_force(tree.traverse(m_data, m_distance_to_node_radius_ratio, v1, mass1));
			update_f(body1, total_force, mass1);
		}
		break;
	case ett_nested_tree:
		tree.traverse(node_visitor);
		break;
	default:
		break;
	}
}

void nbody_engine_simple_bh::print_info() const
{
	nbody_engine_simple::print_info();
	qDebug() << "\tdistance_to_node_radius_ratio:" << m_distance_to_node_radius_ratio;
	qDebug() << "\ttraverse_type:" << (m_traverse_type == ett_cycle ? "cycle" : "nested_tree");
}
