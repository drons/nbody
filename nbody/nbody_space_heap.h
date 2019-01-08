#ifndef NBODY_SPACE_HEAP_H
#define NBODY_SPACE_HEAP_H

#include "nbody_engine_simple_bh.h"
#include "nbody_space_heap_func.h"

class nbody_space_heap : public nbody_heap_func<size_t>
{
protected:
	std::vector<nbvertex_t>	m_mass_center;
	std::vector<nbcoord_t>	m_mass;
	std::vector<nbcoord_t>	m_radius_sqr;
	std::vector<size_t>		m_body_n;
	nbcoord_t				m_distance_to_node_radius_ratio;
public:
	nbody_space_heap();
	void build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
			   const nbcoord_t* mass, nbcoord_t distance_to_node_radius_ratio);

	nbvertex_t traverse(const nbody_data* data, const nbvertex_t& v1, const nbcoord_t mass1) const;
	template<class Visitor>
	void traverse(Visitor visit) const
	{
		for(size_t idx = 0; idx != m_body_n.size(); ++idx)
		{
			size_t	body_n(m_body_n[idx]);
			if(body_n != std::numeric_limits<size_t>::max())
			{
				visit(body_n, m_mass_center[idx], m_mass[idx]);
			}
		}
	}
	const std::vector<nbvertex_t>& get_mass_center() const;
	const std::vector<nbcoord_t>& get_mass() const;
	const std::vector<nbcoord_t>& get_radius_sqr() const;
	const std::vector<size_t>&	get_body_n() const;
private:
	void build(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry,
			   const nbcoord_t* rz, const nbcoord_t* mass, size_t idx, size_t dimension);
};

#endif //NBODY_SPACE_HEAP_H
