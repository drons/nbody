#include "nbody_engine_simple_bh.h"

#include <QDebug>

static constexpr	size_t SPACE_DIMENSIONS = 3;
static constexpr	size_t DIM_NUM_X = 0;
static constexpr	size_t DIM_NUM_Y = 1;
static constexpr	size_t DIM_NUM_Z = 2;
static constexpr	size_t MAX_STACK_SIZE = 64;

class nbody_space_tree
{
	class node
	{
		friend class			nbody_space_tree;
		node*					m_left;
		node*					m_right;
		nbvertex_t				m_mass_center;
		nbcoord_t				m_mass;
		nbcoord_t				m_radius_sqr;
		size_t					m_body_n;
	public:
		explicit node() :
			m_left(nullptr),
			m_right(nullptr),
			m_mass(0),
			m_radius_sqr(0),
			m_body_n(std::numeric_limits<size_t>::max())
		{
		}
		~node()
		{
			delete m_left;
			delete m_right;
		}
		void build(size_t count, size_t* indites,
				   const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				   const nbcoord_t* mass, size_t dimension);
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

		m_root = new node();
		#pragma omp parallel
		#pragma omp single
		m_root->build(count, bodies_indites.data(), rx, ry, rz, mass, 0);
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

void nbody_space_tree::node::build(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry,
								   const nbcoord_t* rz, const nbcoord_t* mass, size_t dimension)
{
	if(count == 1) // It is a leaf
	{
		m_mass_center = nbvertex_t(rx[*indites], ry[*indites], rz[*indites]);
		m_mass = mass[*indites];
		m_body_n = *indites;
		return;
	}

	size_t	left_size = count / 2;
	size_t	right_size = count - left_size;
	size_t*	median = indites + left_size;
	auto comparator_x = [rx](size_t a, size_t b) { return rx[a] < rx[b];};
	auto comparator_y = [ry](size_t a, size_t b) { return ry[a] < ry[b];};
	auto comparator_z = [rz](size_t a, size_t b) { return rz[a] < rz[b];};

	switch(dimension)
	{
	case DIM_NUM_X:
		std::nth_element(indites, median, indites + count, comparator_x);
		break;
	case DIM_NUM_Y:
		std::nth_element(indites, median, indites + count, comparator_y);
		break;
	case DIM_NUM_Z:
		std::nth_element(indites, median, indites + count, comparator_z);
		break;
	default:
		qDebug() << "Unexpected dimension";
		break;
	}

	size_t next_dimension((dimension + 1) % SPACE_DIMENSIONS);
	m_left = new node();
	m_right = new node();

	if(count > NBODY_DATA_BLOCK_SIZE)
	{
		#pragma omp task
		m_left->build(left_size, indites, rx, ry, rz, mass, next_dimension);
		#pragma omp task
		m_right->build(right_size, median, rx, ry, rz, mass, next_dimension);
		#pragma omp taskwait
	}
	else
	{
		m_left->build(left_size, indites, rx, ry, rz, mass, next_dimension);
		m_right->build(right_size, median, rx, ry, rz, mass, next_dimension);
	}

	m_mass = m_left->m_mass + m_right->m_mass;
	m_mass_center = (m_left->m_mass_center * m_left->m_mass +
					 m_right->m_mass_center * m_right->m_mass) / m_mass;
	m_radius_sqr = sqrt(m_left->m_radius_sqr) + sqrt(m_right->m_radius_sqr) +
				   m_left->m_mass_center.distance(m_right->m_mass_center);
	m_radius_sqr = m_radius_sqr * m_radius_sqr;
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
