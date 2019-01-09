#ifndef NBODY_SPACE_HEAP_FUNC_H
#define NBODY_SPACE_HEAP_FUNC_H

#ifndef NB_CALL_TYPE
#define NB_CALL_TYPE
#endif //NB_CALL_TYPE

template<class index_t>
struct nbody_heap_func
{
	static NB_CALL_TYPE index_t left_idx(index_t idx)
	{
		return (idx << 1) + 1;
	}

	static NB_CALL_TYPE index_t rght_idx(index_t idx)
	{
		return (idx << 1) + 2;
	}

	static NB_CALL_TYPE index_t parent_idx(index_t idx)
	{
		return (idx - 1) >> 1;
	}

	static NB_CALL_TYPE bool is_left(index_t idx)
	{
		return idx & 1;
	}

	static NB_CALL_TYPE bool is_right(index_t idx)
	{
		return (idx & 1) == 0;
	}

	static NB_CALL_TYPE index_t left2right(index_t idx)
	{
		return idx + 1;
	}

	static NB_CALL_TYPE index_t next_down(index_t idx)
	{
		// While index is 'right' -> go down
		while(is_right(idx))
		{
			index_t parent = parent_idx(idx);
			// We at root again. Stop traverse.
			if(parent == 0)
			{
				return 0;
			}
			idx = parent;
		}
		return left2right(idx);
	}

	static NB_CALL_TYPE index_t skip_idx(index_t idx)
	{
		// Index is 'left' branch -> next index is 'right'.
		// Simple take next one.
		if(is_left(idx))
		{
			return left2right(idx);
		}
		return next_down(idx);
	}

	static NB_CALL_TYPE index_t next_up(index_t idx, index_t tree_size)
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
