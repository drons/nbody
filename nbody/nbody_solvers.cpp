#include "nbody_solvers.h"

template<class Solver>
static nbody_solver* create_butcher_solver(const QVariantMap& param)
{
	nbody_solver_rk_butcher*	solver = new Solver();

	solver->set_error_threshold(param.value("error_threshold", 1e-4).toDouble());
	solver->set_max_recursion(param.value("max_recursion", 8).toUInt());
	solver->set_refine_steps_count(param.value("refine_steps_count", 1).toUInt());
	solver->set_substep_subdivisions(param.value("substep_subdivisions", 8).toUInt());

	return solver;
}

nbody_solver* nbody_create_solver(const QVariantMap& param)
{
	const QString type(param.value("solver").toString());

	if(type == "adams")
	{
		return new nbody_solver_adams(param.value("rank", 1).toInt());
	}
	else if(type == "euler")
	{
		return new nbody_solver_euler();
	}
	else if(type == "rk4")
	{
		return new nbody_solver_rk4();
	}
	else if(type == "rkck")
	{
		return create_butcher_solver<nbody_solver_rkck>(param);
	}
	else if(type == "rkdp")
	{
		return create_butcher_solver<nbody_solver_rkdp>(param);
	}
	else if(type == "rkf")
	{
		return create_butcher_solver<nbody_solver_rkf>(param);
	}
	else if(type == "rkgl")
	{
		return create_butcher_solver<nbody_solver_rkgl>(param);
	}
	else if(type == "rklc")
	{
		return create_butcher_solver<nbody_solver_rklc>(param);
	}
	else if(type == "trapeze")
	{
		nbody_solver_trapeze* solver =  new nbody_solver_trapeze();
		solver->set_refine_steps_count(param.value("refine_steps_count", 1).toUInt());
		return solver;
	}

	return NULL;
}
