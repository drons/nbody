#ifndef NBODY_SPACE_TREE_H
#define NBODY_SPACE_TREE_H

#include "nbody_engine_simple_bh.h"

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
		explicit node();
		~node();
		void build(size_t count, size_t* indites,
				   const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
				   const nbcoord_t* mass, size_t dimension,
				   nbcoord_t distance_to_node_radius_ratio_sqr);
	};
	node*		m_root;
public:
	nbody_space_tree();
	~nbody_space_tree();

	void build(size_t count, const nbcoord_t* rx, const nbcoord_t* ry, const nbcoord_t* rz,
			   const nbcoord_t* mass, nbcoord_t distance_to_node_radius_ratio);

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
