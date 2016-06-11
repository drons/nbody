#include "nbody_fcompute_opencl.h"
#include <QDebug>
#include <QFile>
#include <QStringList>

#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/cl.hpp>

std::string load_program( const QString& filename )
{
	QFile			file( filename );
	if( !file.open( QFile::ReadOnly ) )
		return std::string();
	std::string	src( file.readAll().data() );
	return src;
}

cl::Program load_programs( const cl::Context& context, const cl::Device& device, const QString& options, const QStringList& files )
{
	std::vector< std::string >	source_data;
	cl::Program::Sources		sources;

	for( int i = 0; i != files.size(); ++i )
	{
		source_data.push_back( load_program( files[i] ) );
		sources.push_back( std::make_pair( source_data.back().data(), source_data.back().size() ) );
	}

	cl::Program	prog( context, sources );

	try
	{
		std::vector<cl::Device>	devices;
		devices.push_back( device );
		prog.build( devices, options.toAscii() );
	}
	catch( cl::Error err )
	{
		qDebug() << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>( device ).data();
		throw err;
	}
	catch(...)
	{
		throw std::exception();
	}
	return prog;
}

/*!
__kernel void ComputeBlock( int offset_n1, int offset_n2,
                            __global const nbcoord_t* vert_x,
                            __global const nbcoord_t* vert_y,
                            __global const nbcoord_t* vert_z,
                            __global const nbcoord_t* mass,
                            __global nbcoord_t* dv_x,
                            __global nbcoord_t* dv_y,
                            __global nbcoord_t* dv_z )
*/
typedef cl::make_kernel< cl_int,cl_int,							//Block offset
						 cl::Buffer&, cl::Buffer&, cl::Buffer&, //vertex_x, vertex_y, vertex_z
						 cl::Buffer&,							//mass
						 cl::Buffer&, cl::Buffer&, cl::Buffer&	//dv_x, dv_y, dv_z
						> ComputeBlock;

struct nbody_fcompute_opencl::data
{
	struct devctx
	{
		cl::Context			context;
		cl::Device			device;
		cl::Program			prog;
		cl::CommandQueue	queue;
		cl::Buffer			vertex_x;
		cl::Buffer			vertex_y;
		cl::Buffer			vertex_z;
		cl::Buffer			mass;
		cl::Buffer			dv_x;
		cl::Buffer			dv_y;
		cl::Buffer			dv_z;
		ComputeBlock		ckernel;

		std::vector<nbcoord_t> host_dv_x;
		std::vector<nbcoord_t> host_dv_y;
		std::vector<nbcoord_t> host_dv_z;

		static QString build_options();
		static QStringList sources();
		devctx( cl::Device& device, size_t count );
	};
	std::vector<devctx>		m_devices;

	void find_devices( size_t count );
	void prepare( devctx&, const nbody_data* data, const nbvertex_t* vertites );
	void compute_block( devctx& ctx, size_t offset_n1, size_t offset_n2, const nbody_data* data );
};

QString nbody_fcompute_opencl::data::devctx::build_options()
{
	QString			options;

	options += "-DNBODY_DATA_BLOCK_SIZE=" + QString::number( NBODY_DATA_BLOCK_SIZE ) + " ";
	options += "-DNBODY_MIN_R=" + QString::number( NBODY_MIN_R ) + " ";
	options += "-Dnbcoord_t=" + QString( nbtype_info<nbcoord_t>::type_name() ) + " ";
	return options;
}

QStringList nbody_fcompute_opencl::data::devctx::sources()
{
	return QStringList() << ":/nbody_fcompute_opencl.cl";
}

nbody_fcompute_opencl::data::devctx::devctx( cl::Device& device, size_t count ) :
	context( device ),
	prog( load_programs( context, device, build_options(), sources() ) ),
	queue( context, device, 0 ),
    vertex_x( context, CL_MEM_READ_ONLY, count*sizeof(nbcoord_t) ),
    vertex_y( context, CL_MEM_READ_ONLY, count*sizeof(nbcoord_t) ),
    vertex_z( context, CL_MEM_READ_ONLY, count*sizeof(nbcoord_t) ),
    mass( context, CL_MEM_READ_ONLY, count*sizeof(nbcoord_t) ),
    dv_x( context, CL_MEM_WRITE_ONLY, count*sizeof(nbcoord_t) ),
    dv_y( context, CL_MEM_WRITE_ONLY, count*sizeof(nbcoord_t) ),
    dv_z( context, CL_MEM_WRITE_ONLY, count*sizeof(nbcoord_t) ),
	ckernel( prog, "ComputeBlockLocal" )
{
}

void nbody_fcompute_opencl::data::find_devices(size_t count)
{
	std::vector<cl::Platform>		platforms;
	cl::Platform::get( &platforms );
	qDebug() << "Available platforms & devices:";
	for( size_t i = 0; i != platforms.size(); ++i )
	{
		const cl::Platform&			platform( platforms[i] );
		std::vector<cl::Device>		devices;

		qDebug() << i << platform.getInfo<CL_PLATFORM_VENDOR>().c_str();

		platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );

		for( size_t j = 0; j != devices.size(); ++j )
		{
			cl::Device&		device( devices[j] );
			qDebug() << "\t--dev_id=" << QString("%1,%2").arg(i).arg(j);
			qDebug() << "\t\t CL_DEVICE_NAME" << device.getInfo<CL_DEVICE_NAME>().c_str();
			qDebug() << "\t\t CL_DEVICE_VERSION" << device.getInfo<CL_DEVICE_VERSION>().c_str();

			m_devices.push_back( devctx( device, count ) );
		}
	}

}

void nbody_fcompute_opencl::data::prepare( devctx& ctx, const nbody_data* data, const nbvertex_t* vertites )
{
	size_t					count = data->get_count();
	const nbcoord_t*		mass = data->get_mass();
	std::vector<nbcoord_t>	tmp;

	tmp.resize( count );
	for( size_t n = 0; n != count; ++n )
		tmp[n] = vertites[n].x;
	ctx.queue.enqueueWriteBuffer( ctx.vertex_x, CL_TRUE, 0, count*sizeof(nbcoord_t), tmp.data() );
	for( size_t n = 0; n != count; ++n )
		tmp[n] = vertites[n].y;
	ctx.queue.enqueueWriteBuffer( ctx.vertex_y, CL_TRUE, 0, count*sizeof(nbcoord_t), tmp.data() );
	for( size_t n = 0; n != count; ++n )
		tmp[n] = vertites[n].z;
	ctx.queue.enqueueWriteBuffer( ctx.vertex_z, CL_TRUE, 0, count*sizeof(nbcoord_t), tmp.data() );

	ctx.queue.enqueueWriteBuffer( ctx.mass, CL_TRUE, 0, count*sizeof(nbcoord_t), mass );

#if defined(CL_VERSION_1_2)
	ctx.queue.enqueueFillBuffer( ctx.dv_x, (nbcoord_t)0.0, 0, count );
	ctx.queue.enqueueFillBuffer( ctx.dv_y, (nbcoord_t)0.0, 0, count );
	ctx.queue.enqueueFillBuffer( ctx.dv_z, (nbcoord_t)0.0, 0, count );
#endif //defined(CL_VERSION_1_2)

	ctx.host_dv_x.resize( count );
	ctx.host_dv_y.resize( count );
	ctx.host_dv_z.resize( count );
}

void nbody_fcompute_opencl::data::compute_block( devctx& ctx, size_t offset_n1, size_t offset_n2, const nbody_data* data )
{
	size_t			count = data->get_count();
	cl::Event		exec_ev;

	cl::NDRange		global_range( count );
	cl::NDRange		local_range( NBODY_DATA_BLOCK_SIZE );
	cl::EnqueueArgs	eargs( ctx.queue, global_range, local_range );

	exec_ev = ctx.ckernel( eargs, offset_n1, offset_n2, ctx.vertex_x, ctx.vertex_y, ctx.vertex_z, ctx.mass, ctx.dv_x, ctx.dv_y, ctx.dv_z );
	exec_ev.wait();

	ctx.queue.enqueueReadBuffer( ctx.dv_x, CL_TRUE, 0, count*sizeof(nbcoord_t), ctx.host_dv_x.data() );
	ctx.queue.enqueueReadBuffer( ctx.dv_y, CL_TRUE, 0, count*sizeof(nbcoord_t), ctx.host_dv_y.data() );
	ctx.queue.enqueueReadBuffer( ctx.dv_z, CL_TRUE, 0, count*sizeof(nbcoord_t), ctx.host_dv_z.data() );
}

nbody_fcompute_opencl::nbody_fcompute_opencl() :
	d( new data() )
{
	info();
}

nbody_fcompute_opencl::~nbody_fcompute_opencl()
{
	delete d;
}

void nbody_fcompute_opencl::fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv )
{
	size_t	count = data->get_count();

	if( d->m_devices.empty() )
	{
		d->find_devices( count );
	}
	if( d->m_devices.empty() )
	{
		qDebug() << Q_FUNC_INFO << "m_devices.empty()";
		return;
	}

	data::devctx& ctx( d->m_devices[0] );
	d->prepare( ctx, data, vertites );
	d->compute_block( ctx, 0, 0, data );

	for( size_t n = 0; n != count; ++n )
	{
		dv[n] = nbvertex_t( ctx.host_dv_x[n], ctx.host_dv_y[n], ctx.host_dv_z[n] );
	}
}


int nbody_fcompute_opencl::info()
{
	std::vector<cl::Platform>		platforms;
	cl::Platform::get( &platforms );
	qDebug() << "Available platforms & devices:";
	for( size_t i = 0; i != platforms.size(); ++i )
	{
		const cl::Platform&			platform( platforms[i] );
		std::vector<cl::Device>		devices;

		qDebug() << i << platform.getInfo<CL_PLATFORM_VENDOR>().c_str();

		platform.getDevices( CL_DEVICE_TYPE_ALL, &devices );

		for( size_t j = 0; j != devices.size(); ++j )
		{
			cl::Device&		device( devices[j] );
			qDebug() << "\t--dev_id=" << QString("%1,%2").arg(i).arg(j);
			qDebug() << "\t\t CL_DEVICE_NAME" << device.getInfo<CL_DEVICE_NAME>().c_str();
			qDebug() << "\t\t CL_DEVICE_TYPE" << device.getInfo<CL_DEVICE_TYPE>();
			qDebug() << "\t\t CL_DRIVER_VERSION" << device.getInfo<CL_DRIVER_VERSION>().c_str();
			qDebug() << "\t\t CL_DEVICE_PROFILE" << device.getInfo<CL_DEVICE_PROFILE>().c_str();
			qDebug() << "\t\t CL_DEVICE_VERSION" << device.getInfo<CL_DEVICE_VERSION>().c_str();
			qDebug() << "\t\t CL_DEVICE_MAX_COMPUTE_UNITS" << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			qDebug() << "\t\t CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
			qDebug() << "\t\t CL_DEVICE_MAX_WORK_GROUP_SIZE" << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
			qDebug() << "\t\t CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT" << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
			qDebug() << "\t\t CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT" << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>();
			qDebug() << "\t\t CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE" << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>();
			qDebug() << "\t\t CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE" << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>();
//			qDebug() << "\t\t CL_DEVICE_GLOBAL_MEM_CACHE_TYPE" << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>().c_str();
			qDebug() << "\t\t CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE" << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
			qDebug() << "\t\t CL_DEVICE_GLOBAL_MEM_CACHE_SIZE" << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
			qDebug() << "\t\t CL_DEVICE_GLOBAL_MEM_SIZE" << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			qDebug() << "\t\t CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE" << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
			qDebug() << "\t\t CL_DEVICE_MAX_CONSTANT_ARGS" << device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>();
			qDebug() << "\t\t CL_DEVICE_LOCAL_MEM_TYPE" << device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
			qDebug() << "\t\t CL_DEVICE_LOCAL_MEM_SIZE" << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

			try
			{
				nbody_fcompute_opencl::data::devctx	c( device, NBODY_DATA_BLOCK_SIZE*1024 );
			}
			catch( cl::Error& e )
			{
				qDebug() << e.err() << e.what();
			}
		}
	}
	return 0;
}
