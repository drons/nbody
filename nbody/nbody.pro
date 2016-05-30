TEMPLATE	= app
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

SOURCES	+= \
	main.cpp \
    nbody_data.cpp \
    nbody_solver.cpp \
	nbody_solver_euler.cpp \
    wgt_nbody_view.cpp
HEADERS	+= \
	summation.h \
    nbody_data.h \
    nbody_solver.h \
	nbody_solver_euler.h \
    wgt_nbody_view.h \
    summation_proxy.h \
    vertex.h

