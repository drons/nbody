#ifndef NBODY_SOLVER_H
#define NBODY_SOLVER_H

#include "nbody_data.h"
#include "nbody_engine.h"

class nbody_data_stream;

class nbody_solver
{
	nbody_engine*						m_engine;
	nbcoord_t							m_min_step;
	nbcoord_t							m_max_step;
public:
	nbody_solver();
	virtual ~nbody_solver();
	void set_engine(nbody_engine*);
	nbody_engine* engine();
	void set_time_step(nbcoord_t min_step, nbcoord_t max_step);
	nbcoord_t get_min_step() const;
	nbcoord_t get_max_step() const;
	int run(nbody_data* data, nbody_data_stream* stream, nbcoord_t max_time, nbcoord_t dump_dt, nbcoord_t check_dt);

	virtual const char* type_name() const = 0;
	virtual void advise(nbcoord_t dt) = 0;
	virtual void print_info() const;
};

#endif // NBODY_SOLVER_H
