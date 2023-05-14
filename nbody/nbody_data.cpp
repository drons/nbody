#include "nbody_data.h"
#include "nbody_engine.h"
#include <qnumeric.h>
#include <QDebug>
#include <QFile>
#include <QTextStream>

#include "summation.h"
#include "summation_proxy.h"

#include <omp.h>
#include <random>

namespace {
constexpr nbcoord_t GravityConst = 1_f; // Gravity constant used in modelling
}

nbody_data::nbody_data() :
	m_count(0),
	m_time(0),
	m_step(0),
	m_box_size(0),
	m_total_mass(0),
	m_density(0),
	m_check_list("PLV"),
	m_initial_kinetic_energy(0),
	m_initial_potential_energy(0),
	m_last_total_kinetic_energy(0),
	m_last_total_potential_energy(0),
	m_initial_values_computed(false),
	m_prev_compute_count(0),
	m_timer_start(omp_get_wtime()),
	m_timer_step(0)
{
}

nbvertex_t nbody_data::force_global(const nbvertex_t& v, nbcoord_t mass) const
{
	nbcoord_t	r2(v.norm());
	if(r2 < m_box_size * m_box_size)
	{
		static const nbcoord_t f(4_f * GravityConst * M_PI / 3_f);
		// Internal sphere volume
		return v * f * mass * m_density;
	}
	// External volume
	return v * ((+GravityConst * mass * m_total_mass) / (r2 * sqrt(r2)));
}

nbvertex_t nbody_data::force(const nbvertex_t& v1, const nbvertex_t& v2, nbcoord_t mass1, nbcoord_t mass2) const
{
	nbvertex_t	dr(v1 - v2);
	nbcoord_t	r2(dr.norm());
	if(r2 < nbody::MinDistance)
	{
		r2 = nbody::MinDistance;
	}
	return dr * ((-GravityConst * mass1 * mass2) / (r2 * sqrt(r2)));
}

nbcoord_t nbody_data::potential_energy(const nbvertex_t* vertites, size_t body1, size_t body2) const
{
	nbvertex_t	dr(vertites[body1] - vertites[body2]);
	nbcoord_t	r2(dr.norm());
	if(r2 < nbody::MinDistance)
	{
		return 0;
	}
	return -(GravityConst * m_mass[body1] * m_mass[body2]) / sqrt(r2);
}

void nbody_data::print_statistics(nbody_engine* engine)
{
	size_t	compute_count = 0;
	if(engine != NULL)
	{
		engine->get_data(this);
		compute_count = engine->get_compute_count();
	}

	double		timer_end = omp_get_wtime();
	nbvertex_t	total_impulce;
	nbvertex_t	total_impulce_moment;
	nbvertex_t	mass_center;
	nbcoord_t	total_kinetic_energy(0);
	nbcoord_t	total_potential_energy(0);

	if(m_check_list.contains("P"))
	{
		total_impulce = summation<nbvertex_t, impulce_proxy>(impulce_proxy(this), m_count);
	}
	if(m_check_list.contains("L"))
	{
		total_impulce_moment = summation<nbvertex_t, impulce_moment_proxy>(impulce_moment_proxy(this), m_count);
	}
	if(m_check_list.contains("E"))
	{
		total_kinetic_energy = summation<nbcoord_t, kinetic_energy_proxy>(kinetic_energy_proxy(this), m_count);
		total_potential_energy = summation<nbcoord_t, potential_energy_proxy>(potential_energy_proxy(this), m_count * m_count);
		total_kinetic_energy /= 2;
		total_potential_energy /= 2;
	}
	if(m_check_list.contains("V"))
	{
		nbcoord_t	total_mass(summation<nbcoord_t, const nbcoord_t*>(m_mass.data(), m_count));
		mass_center = summation<nbvertex_t, mass_center_proxy>(mass_center_proxy(this), m_count);
		mass_center /= total_mass;
	}

	if(!m_initial_values_computed)
	{
		m_initial_impulce = total_impulce;
		m_initial_impulce_moment = total_impulce_moment;
		m_initial_mass_center = mass_center;
		m_initial_kinetic_energy = total_kinetic_energy;
		m_initial_potential_energy = total_potential_energy;
		m_initial_values_computed = true;
	}

	QDebug	g(qDebug());
#if (QT_VERSION >= QT_VERSION_CHECK(5, 3, 0))
	g.noquote();
#endif
	g << "#" << QString("%1").arg(m_step, 8, 10,  QChar('0'))
	  << "t" << QString("%1").arg(m_time, 6, 'f', 6, QChar(' '))
	  << "CC" << QString("%1").arg(compute_count, 8, 10,  QChar('0'))
	  << "SC" << QString("%1").arg(compute_count - m_prev_compute_count, 8, 10,  QChar(' '));

	m_prev_compute_count = compute_count;

	if(m_check_list.contains("P"))
	{
		m_last_total_impulce = total_impulce;
		g << "dP" << QString("%1").arg(get_impulce_err(), 4, 'e', 3);
	}
	if(m_check_list.contains("L"))
	{
		m_last_total_impulce_moment = total_impulce_moment;
		g << "dL" << QString("%1").arg(get_impulce_moment_err(), 4, 'e', 3);
	}
	if(m_check_list.contains("V"))
	{
		m_last_mass_center = mass_center;
		nbvertex_t	mass_center_vel((get_last_mass_center() - get_initial_mass_center()) / m_time);
		g << "Vcm" << QString("%1").arg((mass_center_vel).length(), 4, 'e', 3);
	}
	if(m_check_list.contains("E"))
	{
		m_last_total_kinetic_energy = total_kinetic_energy;
		m_last_total_potential_energy = total_potential_energy;
		g << "dE" << QString("%1").arg(get_energy_err(), 4, 'e', 3);
	}
	double	dt = static_cast<double>(m_step - m_timer_step);
	g << "St" << (timer_end - m_timer_start) / dt
	  << "Wt" << (omp_get_wtime() - m_timer_start) / dt
	  << "";

	m_timer_start = omp_get_wtime();
	m_timer_step = m_step;
}

void nbody_data::dump_body(size_t n)
{
	qDebug() << "#" << n
			 << "M" << m_mass[n]
			 << "R" << m_vertites[n].x << m_vertites[n].y << m_vertites[n].z
			 << "V" << m_velosites[n].x << m_velosites[n].y << m_velosites[n].z;
}

bool nbody_data::resize(size_t s)
{
	m_vertites.resize(s);
	m_velosites.resize(s);
	m_mass.resize(s);
	m_color.resize(s);
	m_radius.resize(s);
	m_count = s;
	return true;
}

void nbody_data::add_body(const nbvertex_t& r, const nbvertex_t& v, const nbcoord_t& m,
						  const nbcolor_t& color, const nbcoord_t& radius)
{
	m_vertites.push_back(r);
	m_velosites.push_back(v);
	m_mass.push_back(m);
	m_color.push_back(color);
	m_radius.push_back(radius);
	++m_count;
}

void nbody_data::advise_time(nbcoord_t dt)
{
	m_time += dt;
	++m_step;
}

nbcoord_t nbody_data::get_time() const
{
	return m_time;
}

void nbody_data::set_time(nbcoord_t t)
{
	m_time = t;
}

size_t nbody_data::get_step() const
{
	return m_step;
}

void nbody_data::set_step(size_t s)
{
	m_step = s;
}

nbvertex_t* nbody_data::get_vertites()
{
	return m_vertites.data();
}

const nbvertex_t* nbody_data::get_vertites() const
{
	return m_vertites.data();
}

nbvertex_t* nbody_data::get_velosites()
{
	return m_velosites.data();
}

const nbvertex_t* nbody_data::get_velosites() const
{
	return m_velosites.data();
}

nbcoord_t* nbody_data::get_mass()
{
	return m_mass.data();
}

const nbcoord_t* nbody_data::get_mass() const
{
	return m_mass.data();
}

nbcolor_t* nbody_data::get_color()
{
	return m_color.data();
}

const nbcolor_t* nbody_data::get_color() const
{
	return m_color.data();
}

size_t nbody_data::get_box_size() const
{
	return m_box_size;
}

size_t nbody_data::get_count() const
{
	return m_count;
}

void nbody_data::set_check_list(const QString& check_list)
{
	m_check_list = check_list;
}

static size_t round_up(size_t n, size_t factor)
{
	return factor * (1 + (n - 1) / factor);
}

void nbody_data::add_galaxy(const nbvertex_t& center, const nbvertex_t& velosity,
							nbcoord_t radius, nbcoord_t total_mass,
							size_t count, const nbcolor_t& color)
{
	std::mt19937_64	rng;
	std::uniform_real_distribution<double> uniform(-radius, radius);
	count = round_up(count, NBODY_DATA_BLOCK_SIZE) - 1;

	nbvertex_t	up(0, 0, 1);
	nbvertex_t	right(1, 0, 0);
	nbcoord_t	black_hole_mass_ratio = 0.999;
	nbcoord_t	black_hole_mass = total_mass * black_hole_mass_ratio;
	nbcoord_t	star_mass = (total_mass - black_hole_mass) / static_cast<nbcoord_t>(count);
	nbcoord_t	all_stars_mass = static_cast<nbcoord_t>(count) * star_mass;

	add_body(center, velosity, black_hole_mass, nbcolor_t(0, 1, 0, 1));

	for(size_t n = 0; n != count; ++n)
	{
		nbvertex_t	r(static_cast<nbcoord_t>(uniform(rng)),
					  static_cast<nbcoord_t>(uniform(rng)),
					  static_cast<nbcoord_t>(uniform(rng)));
		nbcoord_t	rlen(r.length());
		if(rlen > radius)
		{
			--n;
			continue;
		}
		r.z *= 0.3;
		r += center;
		nbvertex_t	v((r - center)^up);
		if(v.length() < 1e-7)
		{
			v = (r - center)^right;
		}
		v.normalize();
		nbcoord_t effective_mass = pow(rlen / radius, 3.0) * all_stars_mass + black_hole_mass;
		v *=  sqrt(force(r, center, star_mass, effective_mass).length() * (r - center).length() / star_mass);
		add_body(r, v + velosity, star_mass, color);
	}
}

void nbody_data::make_universe(size_t star_count, nbcoord_t sx, nbcoord_t sy, nbcoord_t sz)
{
	clear();

	nbcoord_t	radius = sx * 0.5;
	nbcoord_t	galaxy_mass = 1000;
	nbvertex_t	center(sx * 0.5, sy * 0.5, sz * 0.5);
	nbvertex_t	base(radius, 0, 0);
	nbvertex_t	velosity(0, sqrt(force(nbvertex_t(), base, galaxy_mass,
									   galaxy_mass).length() * (base).length() / (2 * galaxy_mass)), 0);
	float		intensity = 0.8f;
	nbcolor_t	color1(intensity * 0.5f, intensity * 0.5f, intensity, 1.0f);
	nbcolor_t	color2(intensity, intensity * 0.5f, intensity * 0.5f, 1.0f);

	add_galaxy(center - base, velosity / 3, radius, galaxy_mass, star_count, color1);
	add_galaxy(center + base, -velosity / 3, radius, galaxy_mass, star_count, color2);
	//add_galaxy( center, vertex_t(), radius, galaxy_mass, star_count );

	m_box_size = static_cast<size_t>(std::max(std::max(static_cast<nbcoord_t>(1), sx), std::max(sy, sz)));
}

void nbody_data::make_uniform_universe(size_t count, nbcoord_t radius)
{
	clear();
	std::mt19937_64	rng;
	std::uniform_real_distribution<double> uniform(-1, 1);
	const nbcolor_t color(1, 1, 1, 1);
	count = round_up(count, NBODY_DATA_BLOCK_SIZE);

	nbcoord_t	star_mass = 1_f;
	nbcoord_t	r2 = radius * radius;
	m_total_mass = 0;
	while(m_count < count)
	{
		nbvertex_t	r(static_cast<nbcoord_t>(uniform(rng)*radius),
					  static_cast<nbcoord_t>(uniform(rng)*radius),
					  static_cast<nbcoord_t>(uniform(rng)*radius));
		if(r.norm() > r2)
		{
			continue;
		}
		add_body(r, nbvertex_t(0, 0, 0), star_mass, color);
		m_total_mass += star_mass;
	}
	m_box_size = static_cast<size_t>(radius);
	m_density = m_total_mass / ((radius * radius * radius) * 4.0 * M_PI / 3.0);
}

nbvertex_t nbody_data::get_initial_impulce() const
{
	return m_initial_impulce;
}

nbvertex_t nbody_data::get_initial_impulce_moment() const
{
	return m_initial_impulce_moment;
}

nbvertex_t nbody_data::get_initial_mass_center() const
{
	return m_initial_mass_center;
}

nbcoord_t nbody_data::get_initial_energy() const
{
	return m_initial_kinetic_energy + m_initial_potential_energy;
}

nbvertex_t nbody_data::get_last_total_impulce() const
{
	return m_last_total_impulce;
}

nbvertex_t nbody_data::get_last_total_impulce_moment() const
{
	return m_last_total_impulce_moment;
}

nbvertex_t nbody_data::get_last_mass_center() const
{
	return m_last_mass_center;
}

nbcoord_t nbody_data::get_last_total_energy() const
{
	return m_last_total_kinetic_energy + m_last_total_potential_energy;
}

nbcoord_t nbody_data::get_impulce_err() const
{
	return fabs(100.0 * (get_last_total_impulce() - get_initial_impulce()).length() /
				get_initial_impulce().length());
}

nbcoord_t nbody_data::get_impulce_moment_err() const
{
	return fabs(100.0 * (get_last_total_impulce_moment() - get_initial_impulce_moment()).length() /
				get_initial_impulce_moment().length());
}

nbcoord_t nbody_data::get_energy_err() const
{
	return fabs(100.0 * (get_last_total_energy() - get_initial_energy()) / get_initial_energy());
}

bool nbody_data::is_equal(const nbody_data& other, nbcoord_t eps) const
{
	if(m_count != other.m_count)
	{
		return false;
	}

	for(size_t n = 0; n != m_count; ++n)
	{
		if(fabs(m_mass[n] - other.m_mass[n]) > eps ||
		   fabs(m_radius[n] - other.m_radius[n]) > eps ||
		   fabs(m_vertites[n].x - other.m_vertites[n].x) > eps ||
		   fabs(m_vertites[n].y - other.m_vertites[n].y) > eps ||
		   fabs(m_vertites[n].z - other.m_vertites[n].z) > eps ||
		   fabs(m_velosites[n].x - other.m_velosites[n].x) > eps ||
		   fabs(m_velosites[n].y - other.m_velosites[n].y) > eps ||
		   fabs(m_velosites[n].z - other.m_velosites[n].z) > eps)
		{
			return false;
		}
	}
	return true;
}

void nbody_data::clear()
{
	m_vertites.clear();
	m_velosites.clear();
	m_mass.clear();
	m_color.clear();
	m_radius.clear();
	m_count = 0;
}

bool nbody_data::save(const QString& fn) const
{
	QFile		file(fn);

	if(!file.open(QFile::WriteOnly))
	{
		qDebug() << "Can't open file" << fn << file.errorString();
		return false;
	}

	QTextStream	s(&file);
	s.setRealNumberPrecision(16);
	s.setRealNumberNotation(QTextStream::ScientificNotation);
	s.setNumberFlags(QTextStream::ForceSign);

	for(size_t n = 0; n != get_count(); ++n)
	{
		s << m_vertites[n].x << " " << m_vertites[n].y << " " << m_vertites[n].z << " "
		  << m_velosites[n].x << " " << m_velosites[n].y << " " << m_velosites[n].z << " "
		  << m_mass[n] << " " << m_radius[n] << "\n";
	}

	return true;
}

bool nbody_data::load(const QString& fn, e_units_type unit_type)
{
	const QString	comment("//");
	QFile			file(fn);

	if(!file.open(QFile::ReadOnly))
	{
		qDebug() << "Can't open file" << fn << file.errorString();
		return false;
	}

	clear();

	nbcoord_t	mass_factor = get_mass_factor(unit_type);
	QTextStream	s(&file);

	for(size_t line_n = 0; !s.atEnd(); ++line_n)
	{
		QString	line(s.readLine());
		int		comment_index(line.indexOf(comment));
		if(comment_index >= 0)
		{
			line = line.mid(0, comment_index);
		}
		line = line.trimmed();
		if(line.isEmpty())
		{
			continue;
		}
		const QStringList	columns(line.split(" ", QString::SkipEmptyParts));

		if(columns.size() < 7)
		{
			qDebug() << "Failed to parse line " << line_n
					 << "<" << line << "> \n"
					 << "Format is 'X Y Z Vx Vy Vz Mass <Radius> <Some other data>',"
					 << " comment is '//'";
			return false;
		}
		nbcoord_t	rvm[7];
		for(int n = 0; n != 7; ++n)
		{
			bool ok = false;
			rvm[n] = columns[n].toDouble(&ok);
			if(!ok)
			{
				qDebug() << "Can't convert column" << n
						 << "to double at line " << line_n
						 << "<" << line << ">";
				return false;
			}
		}
		nbcoord_t	radius  = 0_f;
		if(columns.size() > 7)
		{
			bool ok = false;
			radius = columns[7].toDouble(&ok);
			if(!ok)
			{
				qDebug() << "Can't convert 'radius' column to double at line "
						 << "<" << line << ">";
				return false;
			}
		}
		add_body(nbvertex_t(rvm[0], rvm[1], rvm[2]),
				 nbvertex_t(rvm[3], rvm[4], rvm[5]),
				 rvm[6] * mass_factor, nbcolor_t(1, 1, 1, 1), radius);
	}

	return true;
}

bool nbody_data::load_zeno_ascii(const QString& fn)
{
	QFile			file(fn);
	const size_t	ZENO_DIM = 3;
	if(!file.open(QFile::ReadOnly))
	{
		qDebug() << "Can't open file" << fn << file.errorString();
		return false;
	}

	clear();

	QTextStream	s(&file);
	size_t		count;
	{
		QString line_count(s.readLine());
		bool	ok = false;
		count = static_cast<size_t>(line_count.toULongLong(&ok));
		if(!ok)
		{
			qDebug() << "Can't read body count from" << fn;
			return false;
		}
	}
	{
		QString line_dim(s.readLine());
		bool	ok = false;
		size_t	dim = static_cast<size_t>(line_dim.toULongLong(&ok));
		if(!ok)
		{
			qDebug() << "Can't read body dimensions from" << fn;
			return false;
		}
		if(dim != ZENO_DIM)
		{
			qDebug() << "Invalid dimension count" << dim << "in" << fn;
			return false;
		}
	}
	{
		QString line_skip(s.readLine());
		if(line_skip.isEmpty())
		{
			return false;
		}
	}
	if(!resize(count))
	{
		qDebug() << "Can't resize data to count" << count;
		return 0;
	}

	qDebug() << "Zeno stars count is" << count;

	for(size_t i = 0; i != count; ++i)
	{
		if(s.atEnd())
		{
			qDebug() << "Unexpected data stream end";
			return false;
		}
		QString	mass_line(s.readLine());
		get_mass()[i] = mass_line.toDouble();
	}
	for(size_t i = 0; i != count; ++i)
	{
		if(s.atEnd())
		{
			qDebug() << "Unexpected data stream end";
			return false;
		}
		QStringList	pos_line(s.readLine().split(QChar(' '), QString::SkipEmptyParts));
		if(pos_line.size() != ZENO_DIM)
		{
			qDebug() << "Invalid vector dimensions" << pos_line;
			return false;
		}
		get_vertites()[i] = nbvertex_t(pos_line[0].toDouble(),
									   pos_line[1].toDouble(),
									   pos_line[2].toDouble());
	}

	for(size_t i = 0; i != count; ++i)
	{
		if(s.atEnd())
		{
			qDebug() << "Unexpected data stream end";
			return false;
		}
		QStringList	vel_line(s.readLine().split(QChar(' '), QString::SkipEmptyParts));
		if(vel_line.size() != ZENO_DIM)
		{
			qDebug() << "Invalid vector dimensions" << vel_line;
			return false;
		}
		get_velosites()[i] = nbvertex_t(vel_line[0].toDouble(),
										vel_line[1].toDouble(),
										vel_line[2].toDouble());
	}
	std::fill(m_color.begin(), m_color.end(), nbcolor_t(1, 1, 1, 1));
	m_box_size = 1;
	return true;
}

bool nbody_data::load_initial(const QString& fn, const QString& type)
{
	bool	ret = false;
	if(type.compare("Zeno", Qt::CaseInsensitive) == 0)
	{
		ret = load_zeno_ascii(fn);
	}
	else if(type.compare("G1", Qt::CaseInsensitive) == 0)
	{
		ret = load(fn, eut_G1);
	}
	else if(type.compare("SI", Qt::CaseInsensitive) == 0)
	{
		ret = load(fn, eut_SI);
	}
	else if(type.compare("ADK", Qt::CaseInsensitive) == 0)
	{
		ret = load(fn, eut_au_day_kg);
	}
	else
	{
		qDebug() << "Unknown initial state type. Possible values are: Zeno, G1, SI, ADK.";
		return false;
	}
	if(!ret)
	{
		return false;
	}

	nbvertex_t	r(1_f, 1_f, 1_f);

	for(const auto& v : m_vertites)
	{
		r.x = std::max(v.x, r.x);
		r.y = std::max(v.y, r.y);
		r.z = std::max(v.z, r.z);
	}

	m_box_size = static_cast<size_t>(std::max(r.x, std::max(r.y, r.z)));

	return ret;
}

nbcoord_t nbody_data::get_mass_factor(e_units_type unit_type)
{
	switch(unit_type)
	{
	case eut_G1:
		return 1_f;
	case eut_SI:
		return nbody::MassFactorSI;
	case eut_au_day_kg:
		return nbody::MassFactorAuDayKg;
	}
	//We will never be here
	return 0_f;
}
