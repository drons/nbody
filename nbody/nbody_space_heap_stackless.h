#ifndef NBODY_SPACE_HEAP_STACKLESS_H
#define NBODY_SPACE_HEAP_STACKLESS_H

#include "nbody_space_heap.h"

class nbody_space_heap_stackless : public nbody_space_heap
{
	static size_t parent_idx(size_t idx)
	{
		return (idx - 1) / 2;
	}
	static size_t next_down(size_t idx)
	{
		size_t	parent = parent_idx(idx);
		size_t	rght = rght_idx(parent);
		while(rght == idx)
		{
			// We at root again. Stop traverse.
			if(parent == 0)
			{
				return 0;
			}
			idx = parent;
			parent = parent_idx(idx);
			rght = rght_idx(parent);
		}
		return rght;
	}
	static size_t skip_idx(size_t idx)
	{
		size_t	parent = parent_idx(idx);
		size_t	left = left_idx(parent);
		if(left == idx)
		{
			return rght_idx(parent);
		}
		//qDebug() << "SXX" << idx << parent << left;
		return next_down(idx);
	}
	static size_t next_up(size_t idx, size_t tree_size)
	{
		size_t left = left_idx(idx);
		if(left < tree_size)
		{
			return left;
		}

		return next_down(idx);
	}
public:
	nbvertex_t traverse(const nbody_data* data, const nbvertex_t& v1, const nbcoord_t mass1) const;
	template<class Visitor>
	void traverse(Visitor visit) const
	{
		return nbody_space_heap::traverse(visit);
	}
};

#endif //NBODY_SPACE_HEAP_STACKLESS_H
