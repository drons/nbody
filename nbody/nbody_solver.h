#ifndef NBODY_SOLVER_H
#define NBODY_SOLVER_H

#include <memory>
#include "nbody_data.h"
#include "nbody_engine.h"

class nbody_data_stream;
class nbody_step_visitor;

class NBODY_DLL nbody_solver
{
	nbody_engine*						m_engine;
	nbcoord_t							m_min_step;
	nbcoord_t							m_max_step;
	bool								m_clamp_to_box;

	std::vector<std::shared_ptr<nbody_step_visitor>> m_check_visitors;
	nbody_solver(const nbody_solver&) = delete;
	nbody_solver& operator = (const nbody_solver&) = delete;
public:
	nbody_solver();
	virtual ~nbody_solver();
	void set_engine(nbody_engine*);
	nbody_engine* engine();
	void set_time_step(nbcoord_t min_step, nbcoord_t max_step);
	nbcoord_t get_min_step() const;
	nbcoord_t get_max_step() const;
	void set_clamp_to_box(bool clamp);
	void add_check_visitor(std::shared_ptr<nbody_step_visitor> v);
	int run(nbody_data* data, nbody_data_stream* stream, nbcoord_t max_time, nbcoord_t dump_dt, nbcoord_t check_dt);

	virtual const char* type_name() const = 0;
	virtual void advise(nbcoord_t dt) = 0;
	virtual void print_info() const;
	virtual e_ode_order get_ode_order() const;
	//! Reset solver's state to initial
	virtual void reset();
};

#endif // NBODY_SOLVER_H
