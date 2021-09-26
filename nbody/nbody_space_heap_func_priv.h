#ifndef NBODY_SPACE_HEAP_FUNC_PRIV_H
#define NBODY_SPACE_HEAP_FUNC_PRIV_H

NB_CALL_TYPE index_t left_idx(index_t idx)
{
	return (idx << 1);
}

NB_CALL_TYPE index_t rght_idx(index_t idx)
{
	return (idx << 1) + 1;
}

NB_CALL_TYPE index_t parent_idx(index_t idx)
{
	return (idx) >> 1;
}

NB_CALL_TYPE bool is_left(index_t idx)
{
	return (idx & 1) == 0;
}

NB_CALL_TYPE bool is_right(index_t idx)
{
	return idx & 1;
}

NB_CALL_TYPE index_t left2right(index_t idx)
{
	return idx + 1;
}

NB_CALL_TYPE index_t next_down(index_t idx)
{
// See https://en.wikipedia.org/wiki/Find_first_set
#ifdef __CUDA_ARCH__
	idx = (idx >> (__ffs(~idx) - 1));
#elif defined(__GNUC__) && !defined(__OPENCL_VERSION__)
	idx = idx >> __builtin_ctz(~idx);
#elif __OPENCL_VERSION__ >= 200
	idx = idx >> ctz(~idx);
#else
	// While index is 'right' -> go down
	while(is_right(idx))
	{
		index_t parent = parent_idx(idx);
		// We at root again. Stop traverse.
		if(parent == NBODY_HEAP_ROOT_INDEX)
		{
			return NBODY_HEAP_ROOT_INDEX;
		}
		idx = parent;
	}
#endif //__CUDA_ARCH__
	return left2right(idx);
}

NB_CALL_TYPE index_t skip_idx(index_t idx)
{
	return next_down(idx);
}

NB_CALL_TYPE index_t next_up(index_t idx, index_t tree_size)
{
	index_t left = left_idx(idx);
	if(left < tree_size)
	{
		return left;
	}
	return next_down(idx);
}

#endif //NBODY_SPACE_HEAP_FUNC_PRIV_H

