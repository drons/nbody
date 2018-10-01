#ifndef NBODY_FRAME_COMPRESSOR_H
#define NBODY_FRAME_COMPRESSOR_H

#include <QObject>

class QImage;

class nbody_frame_compressor : public QObject
{
public:
	nbody_frame_compressor();
	virtual ~nbody_frame_compressor();
	virtual bool set_destination(const QString&) = 0;
	virtual void push_frame(const QImage& f, size_t frame_n) = 0;
};

#endif // NBODY_FRAME_COMPRESSOR_H
