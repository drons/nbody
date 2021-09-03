#include "nbody_space_tree.h"

nbody_space_tree::nbody_space_tree() :
	m_root(nullptr)
{
}

nbody_space_tree::~nbody_space_tree()
{
	delete m_root;
}

bool nbody_space_tree::is_empty() const
{
	return m_root == nullptr;
}

nbody_space_tree::node::node() :
	m_left(nullptr),
	m_right(nullptr),
	m_mass(0),
	m_radius_sqr(0),
	m_body_n(TREE_NO_BODY)
{
}

nbody_space_tree::node::~node()
{
	delete m_left;
	delete m_right;
}

void nbody_space_tree::node::update(nbcoord_t distance_to_node_radius_ratio_sqr)
{
	m_mass = m_left->m_mass + m_right->m_mass;
	m_mass_center = (m_left->m_mass_center * m_left->m_mass +
					 m_right->m_mass_center * m_right->m_mass) / m_mass;
	m_bmin = nbvertex_t(std::min(m_left->m_bmin.x, m_right->m_bmin.x),
						std::min(m_left->m_bmin.y, m_right->m_bmin.y),
						std::min(m_left->m_bmin.z, m_right->m_bmin.z));
	m_bmax = nbvertex_t(std::max(m_left->m_bmax.x, m_right->m_bmax.x),
						std::max(m_left->m_bmax.y, m_right->m_bmax.y),
						std::max(m_left->m_bmax.z, m_right->m_bmax.z));
	m_radius_sqr = (m_bmax - m_bmin).length() * static_cast<nbcoord_t>(0.5) +
				   ((m_bmax + m_bmin) / 2 - m_mass_center).length();
	m_radius_sqr = m_radius_sqr * m_radius_sqr * distance_to_node_radius_ratio_sqr;
}

void nbody_space_tree::node::build(size_t count, size_t* indites,
								   const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
								   const nbcoord_t* mass, size_t dimension,
								   nbcoord_t distance_to_node_radius_ratio_sqr)
{
	if(count == 1) // It is a leaf
	{
		m_mass_center = nbvertex_t(rx[*indites], ry[*indites], rz[*indites]);
		m_mass = mass[*indites];
		m_body_n = *indites;
		m_bmin = m_mass_center;
		m_bmax = m_mass_center;
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
		m_left->build(left_size, indites, rx, ry, rz, mass, next_dimension, distance_to_node_radius_ratio_sqr);
		#pragma omp task
		m_right->build(right_size, median, rx, ry, rz, mass, next_dimension, distance_to_node_radius_ratio_sqr);
		#pragma omp taskwait
	}
	else
	{
		m_left->build(left_size, indites, rx, ry, rz, mass, next_dimension, distance_to_node_radius_ratio_sqr);
		m_right->build(right_size, median, rx, ry, rz, mass, next_dimension, distance_to_node_radius_ratio_sqr);
	}

	update(distance_to_node_radius_ratio_sqr);
}

void nbody_space_tree::node::rebuild(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
									 nbcoord_t distance_to_node_radius_ratio_sqr)
{
	if(m_body_n < TREE_NO_BODY) // It is a leaf, update coordinate and box
	{
		m_mass_center = nbvertex_t(rx[m_body_n], ry[m_body_n], rz[m_body_n]);
		m_bmin = m_mass_center;
		m_bmax = m_mass_center;
		return;
	}
	size_t	left_size = count / 2;
	size_t	right_size = count - left_size;

	if(count > NBODY_DATA_BLOCK_SIZE)
	{
		#pragma omp task
		m_left->rebuild(left_size, rx, ry, rz, distance_to_node_radius_ratio_sqr);
		#pragma omp task
		m_right->rebuild(right_size, rx, ry, rz, distance_to_node_radius_ratio_sqr);
		#pragma omp taskwait
	}
	else
	{
		m_left->rebuild(left_size, rx, ry, rz, distance_to_node_radius_ratio_sqr);
		m_right->rebuild(right_size, rx, ry, rz, distance_to_node_radius_ratio_sqr);
	}
	update(distance_to_node_radius_ratio_sqr);
}

nbvertex_t nbody_space_tree::traverse(const nbody_data* data,
									  const nbvertex_t& v1,
									  const nbcoord_t mass1) const
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

		if(distance_sqr > curr->m_radius_sqr)
		{
			total_force += data->force(v1, curr->m_mass_center, mass1, curr->m_mass);
		}
		else
		{
			if(curr->m_right != NULL)
			{
				*stack++ = curr->m_right;
			}
			if(curr->m_left != NULL)
			{
				*stack++ = curr->m_left;
			}
		}
	}
	return total_force;
}

void nbody_space_tree::build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
							 const nbcoord_t* mass, nbcoord_t distance_to_node_radius_ratio)
{
	std::vector<size_t>	bodies_indites;

	bodies_indites.resize(count);
	for(size_t i = 0; i != count; ++i)
	{
		bodies_indites[i] = i;
	}
	delete m_root;
	m_root = new node();
	#pragma omp parallel
	#pragma omp single
	m_root->build(count, bodies_indites.data(), rx, ry, rz, mass, 0,
				  distance_to_node_radius_ratio * distance_to_node_radius_ratio);
}

void nbody_space_tree::rebuild(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
							   nbcoord_t distance_to_node_radius_ratio)
{
	#pragma omp parallel
	#pragma omp single
	m_root->rebuild(count, rx, ry, rz,
					distance_to_node_radius_ratio * distance_to_node_radius_ratio);
}
