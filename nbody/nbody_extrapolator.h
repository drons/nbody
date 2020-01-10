#ifndef NBODY_EXTRAPOLATOR_H
#define NBODY_EXTRAPOLATOR_H

#include "nbody_engine.h"

/*!
	\brief Extrapolate solution to substep_count->inf (dt->0)
*/
class NBODY_DLL nbody_extrapolator
{
	nbody_engine*			m_e;
	std::vector<size_t>		m_sub_steps_count;
public:
	nbody_extrapolator(nbody_engine* engine, size_t order,
					   const std::vector<size_t>& substeps_count);
	virtual ~nbody_extrapolator();
	nbody_engine* engine() const;
	size_t sub_steps_count(size_t level) const;
	virtual void update_table(size_t level, nbody_engine::memory* y) const = 0;
	virtual nbcoord_t estimate_error(size_t level) const = 0;
	virtual void extrapolate(size_t level, nbody_engine::memory* ext_y) const = 0;
};

/*!
	\brief Berrut rational extrapolation

	@see (1.4) at [1]
	[1] J.P. Berrut, R. Baltensperger, H. D. Mittelmann,
		Recent developments in barycentric rational interpolation,
		International Series of Numerical Mathematics Vol. 151., 2005
		http://plato.asu.edu/ftp/papers/paper105.pdf
*/
class NBODY_DLL nbody_extrapolator_berrut : public nbody_extrapolator
{
	nbody_engine::memory_array	m_table;
	nbody_engine::memory*		m_diff;
public:
	nbody_extrapolator_berrut(nbody_engine* engine, size_t order,
							  const std::vector<size_t>& substeps_count);
	~nbody_extrapolator_berrut();
	void update_table(size_t level, nbody_engine::memory* y) const override;
	nbcoord_t estimate_error(size_t level) const override;
	void extrapolate(size_t level, nbody_engine::memory* ext_y) const override;
private:
	std::vector<nbcoord_t> weights(size_t level) const;
};

#endif //NBODY_EXTRAPOLATOR_H
