TEMPLATE	= app
TARGET		= main
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

INCLUDEPATH += ../../nbody
LIBS += -L../../nbody
LIBS += -lnbody

SOURCES	+= main.cpp \
    wgt_nbody_view.cpp

HEADERS	+= \
	wgt_nbody_view.h \



