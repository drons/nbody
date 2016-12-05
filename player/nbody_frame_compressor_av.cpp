#include "nbody_frame_compressor_av.h"

#include <QImage>
#include <QDebug>

extern "C"
{
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>
#include <libavresample/avresample.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>

}

struct nbody_frame_compressor_av::data
{
	//based on https://git.libav.org/?p=libav.git;a=blob;f=doc/examples/output.c;h=bb0da30041cb28c90b65402dd4037f7115555532;hb=HEAD
	AVOutputFormat*		fmt;
	AVFormatContext*	oc;
	AVStream*			st;
	AVCodecContext*		enc;
	AVFrame*			frame;
	SwsContext*			sws_ctx;

	data() :
	    fmt( NULL ),
	    oc( NULL ),
	    st( NULL ),
	    enc( NULL ),
	    frame( NULL ),
	    sws_ctx( NULL )
	{
		static int reg = 0;
		if( reg == 0 )
		{
			av_register_all();
			reg = 1;
		}
	}

	int add_video_stream();
	int open_video();
	void close_stream();
	bool write_frame( const QImage& img );
};

nbody_frame_compressor_av::nbody_frame_compressor_av() : d( new data() )
{
	d->fmt = av_guess_format( "mpeg", NULL, NULL );
	d->oc = avformat_alloc_context();
	d->oc->oformat = d->fmt;

	qDebug() << "d->fmt->video_codec" << d->fmt->video_codec;
}

nbody_frame_compressor_av::~nbody_frame_compressor_av()
{
	d->close_stream();
	delete d;
}

bool nbody_frame_compressor_av::set_destination( const QString& fn )
{
	snprintf( d->oc->filename, sizeof( d->oc->filename ), "%s", fn.toLocal8Bit().data() );

	d->add_video_stream();
	d->open_video();

	av_dump_format( d->oc, 0, d->oc->filename, 1);

	int ret = avio_open( &d->oc->pb, d->oc->filename, AVIO_FLAG_WRITE );
	if (ret < 0)
	{
		qDebug() << "Could not open" << d->oc->filename;
		return false;
	}

	//Magic numbers for d->oc->max_delay are from https://lists.libav.org/pipermail/libav-user/2008-March/000226.html
	static float mux_max_delay= 0.7;
	d->oc->max_delay= (int)(mux_max_delay*AV_TIME_BASE);

	avformat_write_header( d->oc, NULL );

	return true;
}

void nbody_frame_compressor_av::push_frame( const QImage& frame, size_t )
{
	d->write_frame( frame );
}

int nbody_frame_compressor_av::data::add_video_stream()
{
	AVCodec*			codec;
	AVCodecID			codec_id = fmt->video_codec;

	//find the video encoder
	codec = avcodec_find_encoder( codec_id );
	if( codec == NULL )
	{
		qDebug() << "codec not found";
		return -1;
	}

	st = avformat_new_stream( oc, NULL );
	if( st == NULL )
	{
		qDebug() << "Could not alloc stream";
		return -1;
	}

	enc = avcodec_alloc_context3( codec );
	if( enc == NULL )
	{
		qDebug() << "Could not alloc an encoding context";
		return -1;
	}

	enc->rc_buffer_size = 16*1024*1024;
	//Put sample parameters.
	enc->bit_rate = 4000000;
	// Resolution must be a multiple of two.
	enc->width    = 1920;
	enc->height   = 1080;
	//timebase: This is the fundamental unit of time (in seconds) in terms
	// of which frame timestamps are represented. For fixed-fps content,
	// timebase should be 1/framerate and timestamp increments should be
	// identical to 1.
	st->time_base = (AVRational){ 1, 25 };
	enc->time_base = st->time_base;

	enc->gop_size = 12; //emit one intra frame every twelve frames at most
	enc->pix_fmt = AV_PIX_FMT_YUV420P;

	if( enc->codec_id == AV_CODEC_ID_MPEG2VIDEO )
	{
		//just for testing, we also add B-frames
		enc->max_b_frames = 2;
	}

	if( enc->codec_id == AV_CODEC_ID_MPEG1VIDEO )
	{
		// Needed to avoid using macroblocks in which some coeffs overflow.
		// This does not happen with normal video, it just happens here as
		// the motion of the chroma plane does not match the luma plane.
		enc->mb_decision = 2;
	}

	/* Some formats want stream headers to be separate. */
	if( oc->oformat->flags & AVFMT_GLOBALHEADER )
	{
//		c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
	}

	return 0;
}

static AVFrame* alloc_picture( AVPixelFormat pix_fmt, int width, int height )
{
	AVFrame*	picture;
	int ret;

	picture = av_frame_alloc();
	if( picture == NULL )
	{
		return NULL;
	}

	picture->format = pix_fmt;
	picture->width  = width;
	picture->height = height;

	/* allocate the buffers for the frame data */
	ret = av_frame_get_buffer( picture, 32 );
	if( ret < 0 )
	{
		qDebug() << "Could not allocate frame data";
		return NULL;
	}

	return picture;
}

int nbody_frame_compressor_av::data::open_video()
{
	/* open the codec */
	if( avcodec_open2( enc, NULL, NULL) < 0 )
	{
		qDebug() << "could not open codec";
		return -1;
	}

	/* Allocate the encoded raw picture. */
	frame = alloc_picture( enc->pix_fmt, enc->width, enc->height );
	if( frame == NULL )
	{
		qDebug() << "Could not allocate picture\n";
		return -1;
	}

	// copy the stream parameters to the muxer
	avcodec_copy_context( st->codec, enc );

	return 0;
}

void nbody_frame_compressor_av::data::close_stream()
{
	av_write_trailer( oc );

	avcodec_free_context( &enc );
	av_frame_free( &frame );
	sws_freeContext( sws_ctx );

	if ( !( fmt->flags & AVFMT_NOFILE ) )
	{
		avio_close( oc->pb );
	}

	avformat_free_context( oc );
}

bool nbody_frame_compressor_av::data::write_frame( const QImage& img )
{
	int				ret;

	if( sws_ctx == NULL )
	{
		sws_ctx = sws_getContext( img.width(), img.height(), AV_PIX_FMT_BGRA,
								  enc->width, enc->height, enc->pix_fmt,
								  SWS_BICUBIC, NULL, NULL, NULL);
		if( sws_ctx == NULL )
		{
			qDebug() << "Cannot initialize the conversion context";
			return false;
		}
	}
	const uint8_t* const img_data[4] = { img.bits(), img.bits() + 1, img.bits() + 2, img.bits() + 3 };
	const int			 stride[4] = { img.bytesPerLine(), img.bytesPerLine(), img.bytesPerLine(), img.bytesPerLine() };

	ret = av_frame_make_writable( frame );
	if(ret < 0)
	{
		qDebug() << "Cannot av_frame_make_writable";
		return false;
	}

	ret = sws_scale( sws_ctx, img_data, stride,
					 0, img.height(), frame->data, frame->linesize );

	if( ret != img.height() )
	{
		qDebug() << "sws_scale failed" << ret;
		return false;
	}

	int			got_packet = 0;
	AVPacket	pkt = { 0 };

	av_init_packet( &pkt );
	ret = avcodec_encode_video2( enc, &pkt, frame, &got_packet );

	if( ret != 0 )
	{
		qDebug() << "avcodec_encode_video2 failed" << ret;
		return false;
	}

	if( got_packet != 0 )
	{
		av_packet_rescale_ts( &pkt, enc->time_base, st->time_base );
		pkt.stream_index = st->index;
		ret = av_interleaved_write_frame( oc, &pkt );
		if( ret != 0 )
		{
			qDebug() << "av_interleaved_write_frame failed" << ret;
			return false;
		}
	}

	return true;
}
