#ifndef NBODY_BUTCHER_TABLE_H
#define NBODY_BUTCHER_TABLE_H

#include "nbtype.h"

class nbody_butcher_table
{
public:
	nbody_butcher_table();
	virtual ~nbody_butcher_table();
	virtual size_t get_steps() const = 0;
	virtual const nbcoord_t** get_a() const = 0;
	virtual const nbcoord_t* get_b1() const = 0;
	virtual const nbcoord_t* get_b2() const = 0;
	virtual const nbcoord_t* get_c() const = 0;

	virtual bool is_implicit() const = 0;
	virtual bool is_embedded() const = 0;
};

/*!
   \brief Butcher table for classic Runge-Kutta order 4 method
*/
class nbody_butcher_table_rk4 : public nbody_butcher_table
{
public:
	nbody_butcher_table_rk4();

	size_t get_steps() const;
	const nbcoord_t**get_a() const;
	const nbcoord_t*get_b1() const;
	const nbcoord_t*get_b2() const;
	const nbcoord_t*get_c() const;
	bool is_implicit() const;
	bool is_embedded() const;
};

#endif // NBODY_BUTCHER_TABLE_H
