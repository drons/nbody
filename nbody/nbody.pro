include( ../pri/nbody.pri )
include( ../pri/opencl.pri )
include( ../pri/cuda.pri )
include( ../pri/vectorize.pri )

TEMPLATE	= lib
TARGET		= nbody
MOC_DIR = ./.tmp/moc
DESTDIR = ./../lib

CONFIG		+= qt dll
QT += opengl
win32:LIBS += -lGLU32
unix:LIBS += -lGLU

DEFINES += NBODY_EXPORT_DLL

SOURCES	+= \
	nbody_arg_parser.cpp \
	nbody_butcher_table.cpp \
	nbody_data.cpp \
	nbody_engine.cpp \
	nbody_engine_ah.cpp \
	nbody_engine_block.cpp \
	nbody_engine_openmp.cpp \
	nbody_engine_simple.cpp \
	nbody_engine_simple_bh.cpp \
	nbody_engines.cpp \
	nbody_solver.cpp \
	nbody_solver_adams.cpp \
	nbody_solver_euler.cpp \
	nbody_solver_rk_butcher.cpp \
	nbody_solver_rk4.cpp \
	nbody_solver_rkck.cpp \
	nbody_solver_rkdp.cpp \
	nbody_solver_rkdverk.cpp \
	nbody_solver_rkf.cpp \
	nbody_solver_rkgl.cpp \
	nbody_solver_rklc.cpp \
	nbody_solver_stormer.cpp \
	nbody_solver_trapeze.cpp \
	nbody_solvers.cpp \
	nbody_space_heap.cpp \
	nbody_space_heap_stackless.cpp \
	nbody_space_tree.cpp \
	nbody_data_stream.cpp \
	nbody_data_stream_reader.cpp

HEADERS	+= \
	nbody_export.h \
	summation.h \
	summation_proxy.h \
	nbody_arg_parser.h \
	nbody_butcher_table.h \
	nbody_data.h \
	nbody_engine.h \
	nbody_engine_ah.h \
	nbody_engine_block.h \
	nbody_engine_openmp.h \
	nbody_engine_simple.h \
	nbody_engine_simple_bh.h \
	nbody_engines.h \
	nbody_solver.h \
	nbody_solver_adams.h \
	nbody_solver_euler.h \
	nbody_solver_rk_butcher.h \
	nbody_solver_rk4.h \
	nbody_solver_rkck.h \
	nbody_solver_rkdp.h \
	nbody_solver_rkdverk.h \
	nbody_solver_rkf.h \
	nbody_solver_rkgl.h \
	nbody_solver_rklc.h \
	nbody_solver_stormer.h \
	nbody_solver_trapeze.h \
	nbody_solvers.h \
	nbody_space_heap.h \
	nbody_space_heap_stackless.h \
	nbody_space_heap_func.h \
	nbody_space_tree.h \
	vertex.h \
	nbtype.h \
	nbtype_info.h \
	nbody_data_stream.h \
	nbody_data_stream_reader.h

contains( DEFINES, HAVE_OPENCL ){
	OTHER_FILES += nbody_engine_opencl.cl
	RESOURCES += opencl.qrc
	HEADERS += nbody_engine_opencl.h
	SOURCES += nbody_engine_opencl.cpp
	HEADERS += nbody_engine_opencl_bh.h
	SOURCES += nbody_engine_opencl_bh.cpp
}

contains( DEFINES, HAVE_CUDA ){
	HEADERS +=	nbody_engine_cuda.h \
				nbody_engine_cuda_bh.h \
				nbody_engine_cuda_bh_tex.h \
				nbody_engine_cuda_impl.h \
				nbody_engine_cuda_memory.h

	SOURCES +=	nbody_engine_cuda.cpp \
				nbody_engine_cuda_bh.cpp \
				nbody_engine_cuda_bh_tex.cpp \
				nbody_engine_cuda_memory.cpp

	CUSOURCES += nbody_engine_cuda_impl.cu
}

nbodyinst.path = /tmp/nbody
INSTALLS += nbodyinst

nbodydepl.files = *
nbodydepl.path = /tmp/nbody
DEPLOYMENT += nbodydepl



