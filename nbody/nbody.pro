TEMPLATE	= lib
TARGET		= nbody
MOC_DIR = ./.tmp/moc

CONFIG		+= qt
QT += opengl
LIBS += -lGLU -lgomp

NBODY_FLAGS += -fopenmp
NBODY_FLAGS += -ftree-vectorizer-verbose=1
NBODY_FLAGS += -mavx -ffast-math -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -ftree-vectorize -fipa-cp-clone

QMAKE_CFLAGS += $$NBODY_FLAGS
QMAKE_CXXFLAGS += $$NBODY_FLAGS
LIBS += -lOpenCL
LIBS += -L/opt/intel/opencl
INCLUDEPATH += /opt/intel/opencl/include
INCLUDEPATH += /home/sas/prg/opencl/nvidia

SOURCES	+= \
    nbody_butcher_table.cpp \
	nbody_data.cpp \
	nbody_engine.cpp \
	nbody_engine_block.cpp \
	nbody_engine_opencl.cpp \
	nbody_engine_simple.cpp \
#	nbody_engine_sparse.cpp \
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
	nbody_solver_trapeze.cpp

HEADERS	+= \
	summation.h \
	summation_proxy.h \
    nbody_butcher_table.h \
	nbody_data.h \
	nbody_engine.h \
	nbody_engine_block.h \
	nbody_engine_opencl.h \
	nbody_engine_simple.h \
#	nbody_engine_sparse.h \
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
	vertex.h \
    nbtype.h

OTHER_FILES += \
    nbody_engine_opencl.cl

RESOURCES += \
    opencl.qrc


nbodyinst.path = /tmp/nbody
INSTALLS += nbodyinst

nbodydepl.files = *
nbodydepl.path = /tmp/nbody
DEPLOYMENT += nbodydepl



