include( ../pri/nbody.pri )
include( ../pri/opencl.pri )
include( ../pri/opencv.pri )
include( ../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= nbody-player
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt
QT += opengl concurrent
win32:LIBS += -lGLU32 -lopengl32
unix:LIBS += -lGLU

INCLUDEPATH += ../nbody
LIBS += -L../lib
LIBS += -lnbody

SOURCES	+= main.cpp \
    wgt_nbody_view.cpp \
    wgt_nbody_player.cpp \
    wgt_nbody_player_control.cpp \
    nbody_frame_compressor.cpp \
    nbody_frame_compressor_image.cpp \
    nbody_frame_compressor_opencv.cpp

HEADERS	+= \
	wgt_nbody_view.h \
    wgt_nbody_player.h \
    wgt_nbody_player_control.h \
    nbody_frame_compressor.h \
    nbody_frame_compressor_image.h \
    nbody_frame_compressor_opencv.h



