#ifndef NBODY_SPACE_HEAP_H
#define NBODY_SPACE_HEAP_H

#include <vector>
#include "nbody_data.h"
#include "nbody_space_tree.h"
#include "nbody_space_heap_func.h"

class nbody_space_heap : public nbody_heap_func<size_t>
{
protected:
	std::vector<nbvertex_t>	m_mass_center;
	std::vector<nbcoord_t>	m_mass;
	std::vector<nbcoord_t>	m_radius_sqr;
	std::vector<nbvertex_t>	m_box_min;
	std::vector<nbvertex_t>	m_box_max;
	std::vector<size_t>		m_body_n;
	nbcoord_t				m_distance_to_node_radius_ratio_sqr;
public:
	nbody_space_heap();
	//! Check for empty tree
	bool is_empty() const;
	//! Build tree from scratch
	void build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
			   const nbcoord_t* mass, nbcoord_t distance_to_node_radius_ratio);
	//! Rebuild cell boxes, radii and mass centers
	void rebuild(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				 nbcoord_t distance_to_node_radius_ratio);

	nbvertex_t traverse(const nbody_data* data, const nbvertex_t& v1, const nbcoord_t mass1) const;
	template<class Visitor>
	void traverse(Visitor visit) const
	{
		size_t	size = m_body_n.size();
		#pragma omp parallel for schedule(dynamic, 4)
		for(size_t idx = size / 2; idx < size; ++idx)
		{
			size_t	body_n(m_body_n[idx]);
			visit(body_n, m_mass_center[idx], m_mass[idx]);
		}
	}
	const std::vector<nbvertex_t>& get_mass_center() const;
	const std::vector<nbcoord_t>& get_mass() const;
	const std::vector<nbcoord_t>& get_radius_sqr() const;
	const std::vector<size_t>&	get_body_n() const;
private:
	void update(size_t idx, size_t left, size_t rght);
	void build_p(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry,
				 const nbcoord_t* rz, const nbcoord_t* mass, size_t idx, size_t dimension);
};

#endif //NBODY_SPACE_HEAP_H
