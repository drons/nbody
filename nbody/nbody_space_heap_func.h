#ifndef NBODY_SPACE_HEAP_FUNC_H
#define NBODY_SPACE_HEAP_FUNC_H

template<class index_t>
struct nbody_heap_func
{
	static index_t left_idx(index_t idx)
	{
		return 2 * idx + 1;
	}

	static index_t rght_idx(index_t idx)
	{
		return 2 * idx + 2;
	}

	static index_t parent_idx(index_t idx)
	{
		return (idx - 1) / 2;
	}

	static index_t next_down(index_t idx)
	{
		index_t	parent = parent_idx(idx);
		index_t	rght = rght_idx(parent);
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

	static index_t skip_idx(index_t idx)
	{
		index_t	parent = parent_idx(idx);
		index_t	left = left_idx(parent);
		if(left == idx)
		{
			return rght_idx(parent);
		}
		return next_down(idx);
	}

	static index_t next_up(index_t idx, index_t tree_size)
	{
		index_t left = left_idx(idx);
		if(left < tree_size)
		{
			return left;
		}

		return next_down(idx);
	}
};

#endif //NBODY_SPACE_HEAP_FUNC_H
