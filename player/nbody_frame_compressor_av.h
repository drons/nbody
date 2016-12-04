#ifndef NBODY_FRAME_COMPRESSOR_AV_H
#define NBODY_FRAME_COMPRESSOR_AV_H

#include "nbody_frame_compressor.h"

class nbody_frame_compressor_av : public nbody_frame_compressor
{
	struct	data;
	data*	d;
public:
	nbody_frame_compressor_av();
	~nbody_frame_compressor_av();
	bool set_destination( const QString& );
	void push_frame( const QImage& f, size_t frame_n );
};

#endif // NBODY_FRAME_COMPRESSOR_AV_H
