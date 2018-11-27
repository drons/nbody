#include "nbody_space_tree.h"

nbody_space_tree::nbody_space_tree() :
	m_root(nullptr)
{
}

nbody_space_tree::~nbody_space_tree()
{
	delete m_root;
}

nbody_space_tree::node::node() :
	m_left(nullptr),
	m_right(nullptr),
	m_mass(0),
	m_radius_sqr(0),
	m_body_n(std::numeric_limits<size_t>::max())
{
}

nbody_space_tree::node::~node()
{
	delete m_left;
	delete m_right;
}

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

nbvertex_t nbody_space_tree::traverse(const nbody_data* data, nbcoord_t distance_to_node_radius_ratio,
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

void nbody_space_tree::build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
							 const nbcoord_t* mass)
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

