#include "nbody_space_heap.h"

nbody_space_heap::nbody_space_heap() :
	m_distance_to_node_radius_ratio_sqr(0)
{
}

bool nbody_space_heap::is_empty() const
{
	return m_body_n.empty();
}

void nbody_space_heap::build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
							 const nbcoord_t* mass, nbcoord_t distance_to_node_radius_ratio)
{
	std::vector<size_t>	bodies_indites;

	bodies_indites.resize(count);
	for(size_t i = 0; i != count; ++i)
	{
		bodies_indites[i] = i;
	}
	size_t	heap_size = 2 * count - 1 + NBODY_HEAP_ROOT_INDEX;
	m_mass_center.resize(heap_size);
	m_mass.resize(heap_size);
	m_radius_sqr.resize(heap_size);
	m_box_min.resize(heap_size);
	m_box_max.resize(heap_size);
	m_body_n.resize(heap_size);
	std::fill(m_body_n.begin(), m_body_n.end(), TREE_NO_BODY);

	m_distance_to_node_radius_ratio_sqr = distance_to_node_radius_ratio * distance_to_node_radius_ratio;

	#pragma omp parallel
	#pragma omp single
	build_p(count, bodies_indites.data(), rx, ry, rz, mass, NBODY_HEAP_ROOT_INDEX, DIM_NUM_X);
}

void nbody_space_heap::rebuild(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
							   nbcoord_t distance_to_node_radius_ratio)
{
	m_distance_to_node_radius_ratio_sqr = distance_to_node_radius_ratio * distance_to_node_radius_ratio;

	#pragma omp parallel for
	for(size_t idx = count; idx < count * 2; ++idx)
	{
		size_t body_idx = m_body_n[idx];
		m_mass_center[idx] = nbvertex_t(rx[body_idx], ry[body_idx], rz[body_idx]);
		m_box_min[idx] = m_mass_center[idx];
		m_box_max[idx] = m_mass_center[idx];
	}
	for(size_t level_count = count; level_count > 0; level_count /= 2)
	{
		#pragma omp parallel for
		for(size_t idx = level_count / 2; idx < level_count; ++idx)
		{
			size_t	left(left_idx(idx));
			size_t	rght(rght_idx(idx));
			update(idx, left, rght);
		}
	}
}

nbvertex_t nbody_space_heap::traverse(const nbody_data* data, const nbvertex_t& v1, const nbcoord_t mass1) const
{
	nbvertex_t			total_force;

	size_t	stack_data[MAX_STACK_SIZE] = {};
	size_t*	stack = stack_data;
	size_t*	stack_head = stack;

	*stack++ = NBODY_HEAP_ROOT_INDEX;
	while(stack != stack_head)
	{
		size_t				curr = *--stack;
		const nbcoord_t		distance_sqr((v1 - m_mass_center[curr]).norm());

		if(distance_sqr > m_radius_sqr[curr])
		{
			total_force += data->force(v1, m_mass_center[curr], mass1, m_mass[curr]);
		}
		else
		{
			size_t	left(left_idx(curr));
			size_t	rght(rght_idx(curr));
			if(rght < m_body_n.size())
			{
				*stack++ = rght;
			}
			if(left < m_body_n.size())
			{
				*stack++ = left;
			}
		}
	}
	return total_force;
}

const std::vector<nbvertex_t>& nbody_space_heap::get_mass_center() const
{
	return m_mass_center;
}

const std::vector<nbcoord_t>& nbody_space_heap::get_mass() const
{
	return m_mass;
}

const std::vector<nbcoord_t>& nbody_space_heap::get_radius_sqr() const
{
	return m_radius_sqr;
}

const std::vector<size_t>& nbody_space_heap::get_body_n() const
{
	return m_body_n;
}

void nbody_space_heap::update(size_t idx, size_t left, size_t rght)
{
	m_mass[idx] = m_mass[left] + m_mass[rght];
	m_mass_center[idx] = (m_mass_center[left] * m_mass[left] +
						  m_mass_center[rght] * m_mass[rght]) / m_mass[idx];

	m_box_min[idx] = nbvertex_t(std::min(m_box_min[left].x, m_box_min[rght].x),
								std::min(m_box_min[left].y, m_box_min[rght].y),
								std::min(m_box_min[left].z, m_box_min[rght].z));
	m_box_max[idx] = nbvertex_t(std::max(m_box_max[left].x, m_box_max[rght].x),
								std::max(m_box_max[left].y, m_box_max[rght].y),
								std::max(m_box_max[left].z, m_box_max[rght].z));
	nbcoord_t	r = (m_box_max[idx] - m_box_min[idx]).length() * static_cast<nbcoord_t>(0.5) +
					((m_box_max[idx] + m_box_min[idx]) / 2 - m_mass_center[idx]).length();
	m_radius_sqr[idx] = (r * r) * m_distance_to_node_radius_ratio_sqr;
}

void nbody_space_heap::build_p(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry,
							   const nbcoord_t* rz, const nbcoord_t* mass, size_t idx, size_t dimension)
{
	if(count == 1) // It is a leaf
	{
		m_mass_center[idx] = nbvertex_t(rx[*indites], ry[*indites], rz[*indites]);
		m_mass[idx] = mass[*indites];
		m_body_n[idx] = *indites;
		m_box_min[idx] = m_mass_center[idx];
		m_box_max[idx] = m_mass_center[idx];
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

	size_t	next_dimension((dimension + 1) % SPACE_DIMENSIONS);
	size_t	left(left_idx(idx));
	size_t	rght(rght_idx(idx));

	if(count > NBODY_DATA_BLOCK_SIZE)
	{
		#pragma omp task
		build_p(left_size, indites, rx, ry, rz, mass, left, next_dimension);
		#pragma omp task
		build_p(right_size, median, rx, ry, rz, mass, rght, next_dimension);
		#pragma omp taskwait
	}
	else
	{
		build_p(left_size, indites, rx, ry, rz, mass, left, next_dimension);
		build_p(right_size, median, rx, ry, rz, mass, rght, next_dimension);
	}

	update(idx, left, rght);
}
