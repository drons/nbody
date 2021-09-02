#ifndef NBODY_SPACE_TREE_H
#define NBODY_SPACE_TREE_H

#include "nbody_data.h"

static constexpr	size_t SPACE_DIMENSIONS = 3;
static constexpr	size_t DIM_NUM_X = 0;
static constexpr	size_t DIM_NUM_Y = 1;
static constexpr	size_t DIM_NUM_Z = 2;
static constexpr	size_t MAX_STACK_SIZE = 64;
static constexpr	size_t TREE_NO_BODY = std::numeric_limits<size_t>::max();

class nbody_space_tree
{
	class node
	{
		friend class			nbody_space_tree;
		node*					m_left;
		node*					m_right;
		nbvertex_t				m_mass_center;
		nbcoord_t				m_mass;
		nbcoord_t				m_radius_sqr;
		nbvertex_t				m_bmin;
		nbvertex_t				m_bmax;
		size_t					m_body_n;
	public:
		node();
		~node();
		void build(size_t count, size_t* indites,
				   const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				   const nbcoord_t* mass, size_t dimension,
				   nbcoord_t distance_to_node_radius_ratio_sqr);
		void rebuild(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
					 nbcoord_t distance_to_node_radius_ratio_sqr);
		void update(nbcoord_t distance_to_node_radius_ratio_sqr);
	};
	node*		m_root;
public:
	nbody_space_tree();
	~nbody_space_tree();

	//! Check for empty tree
	bool is_empty() const;
	//! Build tree from scratch
	void build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
			   const nbcoord_t* mass, nbcoord_t distance_to_node_radius_ratio);
	//! Rebuild cell boxes, radii and mass centers
	void rebuild(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				 nbcoord_t distance_to_node_radius_ratio);

	template<class Visitor>
	void traverse(Visitor visit) const
	{
		node*	stack_data[MAX_STACK_SIZE] = {};
		node**	stack = stack_data;
		node**	stack_head = stack;

		*stack++ = m_root;
		while(stack != stack_head)
		{
			node*				curr = *--stack;
			if(curr->m_radius_sqr > 0)
			{
				if(curr->m_left != NULL)
				{
					*stack++ = curr->m_left;
				}
				if(curr->m_right != NULL)
				{
					*stack++ = curr->m_right;
				}
			}
			else
			{
				visit(curr->m_body_n, curr->m_mass_center, curr->m_mass);
			}
		}
	}

	nbvertex_t traverse(const nbody_data* data, const nbvertex_t& v1, const nbcoord_t mass1) const;
};



#endif //NBODY_SPACE_TREE_H
