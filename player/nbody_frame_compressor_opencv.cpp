#include "nbody_frame_compressor_opencv.h"

#include <opencv2/opencv.hpp>
#include <QImage>
#include <QDebug>


struct nbody_frame_compressor_opencv::data
{
	cv::VideoWriter m_writer;
	QString			m_dst_file;
};

nbody_frame_compressor_opencv::nbody_frame_compressor_opencv() : d(new data())
{
}

nbody_frame_compressor_opencv::~nbody_frame_compressor_opencv()
{
	delete d;
}

bool nbody_frame_compressor_opencv::set_destination(const QString& fn)
{
	d->m_dst_file = fn;
	return true;
}

void nbody_frame_compressor_opencv::push_frame(const QImage& frame, size_t)
{
	cv::Size    size(frame.width(), frame.height());

	if((!d->m_writer.isOpened()) &&
			(!d->m_writer.open(d->m_dst_file.toLocal8Bit().data(), CV_FOURCC('M', 'P', 'E', 'G'), 24, size, true)))
	{
		qDebug() << "Can't open output" << d->m_dst_file;
		return;
	}

	QImage			rgb(frame.convertToFormat(QImage::Format_RGB888).rgbSwapped());
	const cv::Mat	img(size, CV_8UC3, const_cast<uchar*>(rgb.bits()));

	d->m_writer.write(img);
}
