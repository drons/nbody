#include "nbody_engine_simple_bh.h"

#include <memory>
#include <QDebug>

#include "summation.h"

class nbody_space_tree
{
	class node
	{
		static constexpr		size_t MIN_NODE_BODIES = 4;
		static constexpr		size_t SPACE_DIMENSIONS = 3;
		size_t					m_dimension;
		node*					m_left;
		node*					m_right;
		std::vector<size_t>		m_bodies_indites;
		nbvertex_t				m_mass_center;
		nbvertex_t				m_min;
		nbvertex_t				m_max;
		nbcoord_t				m_mass;
		nbcoord_t				m_radius_sqr;
		size_t					m_count;
	public:
		node(size_t dim = 0) :
			m_dimension(dim),
			m_left(nullptr),
			m_right(nullptr),
			m_min(std::numeric_limits<nbcoord_t>::max(),
				  std::numeric_limits<nbcoord_t>::max(),
				  std::numeric_limits<nbcoord_t>::max()),
			m_max(-std::numeric_limits<nbcoord_t>::max(),
				  -std::numeric_limits<nbcoord_t>::max(),
				  -std::numeric_limits<nbcoord_t>::max()),
			m_mass(0),
			m_radius_sqr(0),
			m_count(0)
		{
		}
		~node()
		{
			delete m_left;
			delete m_right;
		}
		void build(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				   const nbcoord_t* mass);
		nbvertex_t traverse(size_t body1, const nbody_data* data,
							const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz, const nbcoord_t* mass) const;
	};

	node*	m_root;

public:

	void build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz, const nbcoord_t* mass)
	{
		std::vector<size_t>	bodies_indites;

		bodies_indites.resize(count);
		for(size_t i = 0; i != count; ++i)
		{
			bodies_indites[i] = i;
		}

		m_root = new node(0);
		m_root->build(count, bodies_indites.data(), rx, ry, rz, mass);
	}

	nbvertex_t traverse(size_t body1, const nbody_data* data,
						const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz, const nbcoord_t* mass) const
	{
		return m_root->traverse(body1, data, rx, ry, rz, mass);
	}
};

struct mass_center_mass_proxy
{
	const size_t* indites;
	const nbcoord_t* rx;
	const nbcoord_t* ry;
	const nbcoord_t* rz;
	const nbcoord_t* mass;
public:
	mass_center_mass_proxy(size_t* _indites,
						   const nbcoord_t* _rx, const nbcoord_t* _ry, const nbcoord_t* _rz,
						   const nbcoord_t* _mass):
		indites(_indites),
		rx(_rx),
		ry(_ry),
		rz(_rz),
		mass(_mass)
	{
	}
	vertex4<nbcoord_t> operator [](size_t n) const
	{
		size_t	i(indites[n]);
		return vertex4<nbcoord_t>(rx[i] * mass[i],
								  ry[i] * mass[i],
								  rz[i] * mass[i],
								  mass[i]);
	}
};

void nbody_space_tree::node::build(size_t count, size_t* indites, const nbcoord_t* rx, const nbcoord_t* ry,
								   const nbcoord_t* rz, const nbcoord_t* mass)
{
	vertex4<nbcoord_t>	mass_center_mass(summation<vertex4<nbcoord_t>, mass_center_mass_proxy>(
											 mass_center_mass_proxy(indites, rx, ry, rz, mass), count));

	m_mass_center.x = mass_center_mass.x / mass_center_mass.w;
	m_mass_center.y = mass_center_mass.y / mass_center_mass.w;
	m_mass_center.z = mass_center_mass.z / mass_center_mass.w;
	m_mass = mass_center_mass.w;

	nbcoord_t	max_rad_sqr = 0;
	for(size_t n = 0; n != count; ++n)
	{
		size_t				i(indites[n]);
		const nbvertex_t	v(rx[i], ry[i], rz[i]);

		nbcoord_t			rad_sqr((m_mass_center - v).norm());
		max_rad_sqr = std::max(max_rad_sqr, rad_sqr);

		for(size_t d = 0; d != SPACE_DIMENSIONS; ++d)
		{
			if(m_max[d] < v[d])
			{
				m_max[d] = v[d];
			}
			else if(m_min[d] > v[d])
			{
				m_min[d] = v[d];
			}
		}
	}

	m_radius_sqr = max_rad_sqr;
	m_count = count;

	if(count <= MIN_NODE_BODIES)
	{
		m_bodies_indites.resize(count);
		for(size_t n = 0; n != count; ++n)
		{
			m_bodies_indites[n] = indites[n];
		}
	}
	else
	{
		std::vector<size_t>	left, right;
		left.reserve(count / 2);
		right.reserve(count / 2);

		for(size_t n = 0; n != count; ++n)
		{
			size_t				i(indites[n]);
			const nbvertex_t	v(rx[i], ry[i], rz[i]);
			if(v[m_dimension] < m_mass_center[m_dimension])
			{
				left.push_back(i);
			}
			else
			{
				right.push_back(i);
			}
		}

		size_t next_dimension((m_dimension + 1) % SPACE_DIMENSIONS);
		m_left = new node(next_dimension);
		m_right = new node(next_dimension);

		m_left->build(left.size(), left.data(), rx, ry, rz, mass);
		m_right->build(right.size(), right.data(), rx, ry, rz, mass);
	}
}

nbvertex_t nbody_space_tree::node::traverse(size_t body1, const nbody_data* data,
											const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
											const nbcoord_t* mass) const
{
	const nbvertex_t	v1(rx[body1], ry[body1], rz[body1]);
	const nbcoord_t		distance2((v1 - m_mass_center).norm());
	constexpr nbcoord_t	ratio = 50;

	if(m_bodies_indites.empty())
	{
		if(distance2 / m_radius_sqr > ratio)
		{
			//qDebug() << body1;
			return data->force(v1, m_mass_center, mass[body1], m_mass);
		}
		else
		{
			nbvertex_t	total_force;
			if(m_left != NULL)
			{
				total_force += m_left->traverse(body1, data, rx, ry, rz, mass);
			}
			if(m_right != NULL)
			{
				total_force += m_right->traverse(body1, data, rx, ry, rz, mass);
			}
			return total_force;
		}
	}
	else
	{
		nbvertex_t			total_force;
		const nbcoord_t		mass1(mass[body1]);

		for(size_t n = 0; n != m_bodies_indites.size(); ++n)
		{
			size_t body2(m_bodies_indites[n]);

			if(body1 == body2)
			{
				continue;
			}

			const nbvertex_t	v2(rx[ body2 ], ry[ body2 ], rz[ body2 ]);
			const nbvertex_t	f(data->force(v1, v2, mass1, mass[body2]));
			total_force += f;
		}
		return total_force;
	}
}

nbody_engine_simple_bh::nbody_engine_simple_bh()
{
}

const char* nbody_engine_simple_bh::type_name() const
{
	return "nbody_engine_simple_bh";
}

void nbody_engine_simple_bh::fcompute(const nbcoord_t& t, const memory* _y, memory* _f, size_t yoff, size_t foff)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
	smemory*		f = dynamic_cast<smemory*>(_f);

	if(y == NULL)
	{
		qDebug() << "y is not smemory";
		return;
	}
	if(f == NULL)
	{
		qDebug() << "f is not smemory";
		return;
	}

	advise_compute_count();

	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = reinterpret_cast<const nbcoord_t*>(y->data()) + yoff;
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	vx = rx + 3 * count;
	const nbcoord_t*	vy = rx + 4 * count;
	const nbcoord_t*	vz = rx + 5 * count;

	nbcoord_t*			frx = reinterpret_cast<nbcoord_t*>(f->data()) + foff;
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2 * count;
	nbcoord_t*			fvx = frx + 3 * count;
	nbcoord_t*			fvy = frx + 4 * count;
	nbcoord_t*			fvz = frx + 5 * count;

	const nbcoord_t*	mass = reinterpret_cast<const nbcoord_t*>(m_mass->data());

	nbody_space_tree	tree;

	tree.build(count, rx, ry, rz, mass);

	for(size_t body1 = 0; body1 < count; ++body1)
	{
		nbvertex_t			total_force(tree.traverse(body1, m_data, rx, ry, rz, mass));

		frx[body1] = vx[body1];
		fry[body1] = vy[body1];
		frz[body1] = vz[body1];
		fvx[body1] = total_force.x / mass[body1];
		fvy[body1] = total_force.y / mass[body1];
		fvz[body1] = total_force.z / mass[body1];
	}
}

