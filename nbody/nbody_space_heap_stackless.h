#ifndef NBODY_SPACE_HEAP_STACKLESS_H
#define NBODY_SPACE_HEAP_STACKLESS_H

#include "nbody_space_heap.h"

class nbody_space_heap_stackless : public nbody_space_heap
{
public:
	nbvertex_t traverse(const nbody_data* data, const nbvertex_t& v1, const nbcoord_t mass1) const;
	template<class Visitor>
	void traverse(Visitor visit) const
	{
		return nbody_space_heap::traverse(visit);
	}
};

#endif //NBODY_SPACE_HEAP_STACKLESS_H
