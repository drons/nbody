CONFIG += depend_includepath

contains( CONFIG, build-gcov ){
	QMAKE_CFLAGS += -fprofile-arcs -ftest-coverage
	QMAKE_CXXFLAGS += -fprofile-arcs -ftest-coverage
	QMAKE_LFLAGS += -lgcov --coverage
}
