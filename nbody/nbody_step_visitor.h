#ifndef NBODY_STEP_VISITOR_H
#define NBODY_STEP_VISITOR_H

class nbody_data;
class nbody_engine;

class nbody_step_visitor
{
public:
	virtual ~nbody_step_visitor() {};
	virtual void visit(const nbody_data*) = 0;
};

#endif //NBODY_STEP_VISITOR_H
