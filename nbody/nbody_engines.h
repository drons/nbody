#ifndef NBODY_ENGINES_H
#define NBODY_ENGINES_H

#include "nbody_engine_ah.h"
#include "nbody_engine_block.h"
#include "nbody_engine_opencl.h"
#include "nbody_engine_opencl_bh.h"
#include "nbody_engine_openmp.h"
#include "nbody_engine_simple.h"
#include "nbody_engine_simple_bh.h"

/*!
   \brief Create solver from parameters
   \param param - engine type and parameters
   \return configured engine
 */
nbody_engine NBODY_DLL* nbody_create_engine(const QVariantMap& param);

#endif //NBODY_ENGINES_H
