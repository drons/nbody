#include "nbody_space_heap_stackless.h"

nbvertex_t nbody_space_heap_stackless::traverse(const nbody_data* data, const nbvertex_t& v1,
												const nbcoord_t mass1) const
{
	nbvertex_t	total_force;
	size_t		curr = NBODY_HEAP_ROOT_INDEX;
	size_t		tree_size = m_mass_center.size();

	do
	{
		Q_ASSERT(curr < tree_size);
		const nbcoord_t		distance_sqr((v1 - m_mass_center[curr]).norm());

		if(distance_sqr > m_radius_sqr[curr])
		{
			total_force += data->force(v1, m_mass_center[curr], mass1, m_mass[curr]);
			curr = skip_idx(curr);
		}
		else
		{
			curr = next_up(curr, tree_size);
		}
	}
	while(curr != NBODY_HEAP_ROOT_INDEX);

	return total_force;
}
