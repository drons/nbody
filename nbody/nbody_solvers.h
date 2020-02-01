#ifndef NBODY_SOLVERS_H
#define NBODY_SOLVERS_H

#include "nbody_solver_adams.h"
#include "nbody_solver_bulirsch_stoer.h"
#include "nbody_solver_euler.h"
#include "nbody_solver_midpoint.h"
#include "nbody_solver_midpoint_stetter.h"
#include "nbody_solver_rk4.h"
#include "nbody_solver_rkck.h"
#include "nbody_solver_rkdp.h"
#include "nbody_solver_rkdverk.h"
#include "nbody_solver_rkf.h"
#include "nbody_solver_rkfeagin10.h"
#include "nbody_solver_rkfeagin12.h"
#include "nbody_solver_rkfeagin14.h"
#include "nbody_solver_rkgl.h"
#include "nbody_solver_rklc.h"
#include "nbody_solver_stormer.h"
#include "nbody_solver_trapeze.h"

/*!
   \brief Create solver from parameters
   \param param - solver type and parameters
   \return configured solver
 */
nbody_solver NBODY_DLL* nbody_create_solver(const QVariantMap& param);

#endif //NBODY_SOLVERS_H
