#ifndef NBODY_DATA_H
#define NBODY_DATA_H

#include <vector>
#include "nbtype_info.h"
#include "nbody_export.h"

class nbody_engine;

class NBODY_DLL nbody_data
{
	size_t						m_count;
	nbcoord_t					m_time;
	size_t						m_step;
	size_t						m_box_size;
	nbcoord_t					m_total_mass;
	nbcoord_t					m_density; // Mean mass density in box
	std::vector< nbvertex_t >	m_vertites;
	std::vector< nbcolor_t >	m_color;
	std::vector< nbvertex_t >	m_velosites;
	std::vector< nbcoord_t >	m_mass;
	std::vector< nbcoord_t >	m_radius;

	QString						m_check_list;
	nbvertex_t					m_initial_impulce;
	nbvertex_t					m_initial_impulce_moment;
	nbvertex_t					m_initial_mass_center;
	nbcoord_t					m_initial_kinetic_energy;
	nbcoord_t					m_initial_potential_energy;
	nbvertex_t					m_last_total_impulce;
	nbvertex_t					m_last_total_impulce_moment;
	nbvertex_t					m_last_mass_center;
	nbcoord_t					m_last_total_kinetic_energy;
	nbcoord_t					m_last_total_potential_energy;
	bool						m_initial_values_computed;
	size_t						m_prev_compute_count;
	double						m_timer_start;
	size_t						m_timer_step;

public:
	nbody_data();

	nbvertex_t force_global(const nbvertex_t& v, nbcoord_t mass) const;
	nbvertex_t force(const nbvertex_t& v1, const nbvertex_t& v2, nbcoord_t mass1, nbcoord_t mass2) const;
	nbcoord_t potential_energy(const nbvertex_t* vertites, size_t body1, size_t body2) const;
	void add_body(const nbvertex_t& r, const nbvertex_t& v, const nbcoord_t& m,
				  const nbcolor_t& color, const nbcoord_t& radius = 0_f);
	void advise_time(nbcoord_t dt);
	nbcoord_t get_time() const;
	void set_time(nbcoord_t t);
	size_t get_step() const;
	void set_step(size_t s);
	nbvertex_t* get_vertites();
	const nbvertex_t* get_vertites() const;
	nbvertex_t* get_velosites();
	const nbvertex_t* get_velosites() const;
	nbcoord_t* get_mass();
	const nbcoord_t* get_mass() const;
	nbcolor_t* get_color();
	const nbcolor_t* get_color() const;
	size_t get_box_size() const;
	void print_statistics(nbody_engine*);
	void dump_body(size_t n);
	bool resize(size_t);
	size_t get_count() const;

	//! List of fundamental laws of physics to check
	void set_check_list(const QString& check_list);

	void add_galaxy(const nbvertex_t& center, const nbvertex_t& velosity,
					nbcoord_t radius, nbcoord_t total_mass, size_t count,
					const nbcolor_t& color);
	void make_universe(size_t star_count, nbcoord_t sx, nbcoord_t sy, nbcoord_t sz);
	void make_uniform_universe(size_t star_count, nbcoord_t radius);

	nbvertex_t get_initial_impulce() const;
	nbvertex_t get_initial_impulce_moment() const;
	nbvertex_t get_initial_mass_center() const;
	nbcoord_t get_initial_energy() const;
	nbvertex_t get_last_total_impulce() const;
	nbvertex_t get_last_total_impulce_moment() const;
	nbvertex_t get_last_mass_center() const;
	nbcoord_t get_last_total_energy() const;

	nbcoord_t get_impulce_err() const;
	nbcoord_t get_impulce_moment_err() const;
	nbcoord_t get_energy_err() const;

	bool is_equal(const nbody_data& other, nbcoord_t eps = 0) const;
	void clear();
	bool save(const QString& fn) const;
	bool load(const QString& fn, e_units_type unit_type = eut_G1);
	bool load_zeno_ascii(const QString& fn);
	bool load_initial(const QString& fn, const QString& type);

	static nbcoord_t get_mass_factor(e_units_type unit_type);
};

#endif // NBODY_DATA_H
