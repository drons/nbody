#ifndef NBODY_FRAME_COMPRESSOR_OPENCV_H
#define NBODY_FRAME_COMPRESSOR_OPENCV_H

#include "nbody_frame_compressor.h"

class nbody_frame_compressor_opencv : public nbody_frame_compressor
{
	struct	data;
	data*	d;
public:
	nbody_frame_compressor_opencv();
	~nbody_frame_compressor_opencv();
	bool set_destination(const QString&) override;
	void push_frame(const QImage& f, size_t frame_n) override;
};

#endif // NBODY_FRAME_COMPRESSOR_OPENCV_H
