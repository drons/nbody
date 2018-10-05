include( ../pri/nbody.pri )
include( ../pri/opencl.pri )
include( ../pri/vectorize.pri )

TEMPLATE	= lib
TARGET		= nbody
MOC_DIR = ./.tmp/moc

CONFIG		+= qt
QT += opengl
LIBS += -lGLU

SOURCES	+= \
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
	nbody_solver_rkf.cpp \
	nbody_solver_rkgl.cpp \
    nbody_solver_rklc.cpp \
	nbody_solver_stormer.cpp \
	nbody_solver_trapeze.cpp \
	nbody_solvers.cpp \
    nbody_data_stream.cpp \
    nbody_data_stream_reader.cpp

HEADERS	+= \
	summation.h \
	summation_proxy.h \
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
	nbody_solver_rkf.h \
	nbody_solver_rkgl.h \
	nbody_solver_rklc.h \
	nbody_solver_stormer.h \
	nbody_solver_trapeze.h \
	nbody_solvers.h \
	vertex.h \
    nbtype.h \
    nbody_data_stream.h \
    nbody_data_stream_reader.h

contains( DEFINES, HAVE_OPENCL ){
	OTHER_FILES += nbody_engine_opencl.cl
	RESOURCES += opencl.qrc
	HEADERS += nbody_engine_opencl.h
	SOURCES += nbody_engine_opencl.cpp
}

nbodyinst.path = /tmp/nbody
INSTALLS += nbodyinst

nbodydepl.files = *
nbodydepl.path = /tmp/nbody
DEPLOYMENT += nbodydepl



