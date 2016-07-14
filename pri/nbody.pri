CONFIG += depend_includepath

contains( CONFIG, build-gcov ){
	QMAKE_CFLAGS += -fprofile-arcs -ftest-coverage
	QMAKE_CXXFLAGS += -fprofile-arcs -ftest-coverage
	QMAKE_LFLAGS += -lgcov --coverage
}

QMAKE_CXXFLAGS += -Wreorder
QMAKE_CXXFLAGS += -Wunused-variable
QMAKE_CXXFLAGS += -Wall
QMAKE_CXXFLAGS += -Wextra
#QMAKE_CXXFLAGS += -Weffc++
QMAKE_CXXFLAGS += -Woverloaded-virtual
QMAKE_CXXFLAGS += -Wnon-virtual-dtor
QMAKE_CXXFLAGS += -Winit-self
QMAKE_CXXFLAGS += -Wunreachable-code
QMAKE_CXXFLAGS += -Wsequence-point
QMAKE_CXXFLAGS += -Wuninitialized
