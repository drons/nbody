#ifndef NBODY_SPACE_HEAP_FUNC_H
#define NBODY_SPACE_HEAP_FUNC_H

#ifndef NB_CALL_TYPE
#define NB_CALL_TYPE static
#endif //NB_CALL_TYPE

#define NBODY_HEAP_ROOT_INDEX 1

template<class index_t>
struct nbody_heap_func
{
#include "nbody_space_heap_func_priv.h"
};

#endif //NBODY_SPACE_HEAP_FUNC_H
