QMAKE_CXXFLAGS += -std=c++11
QMAKE_CXXFLAGS += -Werror=all
QMAKE_CXXFLAGS += -Werror=extra
QMAKE_CXXFLAGS += -Werror=unused
QMAKE_CXXFLAGS += -Werror=pointer-arith
QMAKE_CXXFLAGS += -Werror=overloaded-virtual
QMAKE_CXXFLAGS += -Werror=non-virtual-dtor
#QMAKE_CXXFLAGS += -Wsign-conversion
QMAKE_CXXFLAGS += -Werror=init-self
QMAKE_CXXFLAGS += -Werror=unreachable-code
QMAKE_CXXFLAGS += -Werror=sequence-point
QMAKE_CXXFLAGS += -Werror=enum-compare
QMAKE_CXXFLAGS += -Werror=uninitialized
QMAKE_CXXFLAGS += -Werror=cast-qual
QMAKE_CXXFLAGS += -Werror=vla
QMAKE_CXXFLAGS += -Werror=format-security
QMAKE_CXXFLAGS += -Werror=format
QMAKE_CXXFLAGS += -Werror=logical-op
QMAKE_CXXFLAGS += -fpermissive
QMAKE_CXXFLAGS += -Werror=maybe-uninitialized
#QMAKE_CXXFLAGS += -Werror=switch-enum

greaterThan(QT_MAJOR_VERSION,4){
#	QMAKE_CXXFLAGS += -Weffc++
#	QMAKE_CXXFLAGS += -Werror=conversion
}

clang {
	#QMAKE_CXXFLAGS += -Werror=shorten-64-to-32 # Errors in Qt5
	QMAKE_CXXFLAGS += -Werror=unused-private-field
	QMAKE_CXXFLAGS += -Wno-inconsistent-missing-override
}

COMPILER_VERSION = $$system($$QMAKE_CXX " -dumpversion")
COMPILER_VERSIONS_LIST=$$split(COMPILER_VERSION, '.')
COMPILER_MAJOR_VERSION = $$first(COMPILER_VERSIONS_LIST)

greaterThan(COMPILER_MAJOR_VERSION, 4){
	# GCC 5+
	QMAKE_CXXFLAGS += -Werror=logical-not-parentheses
	QMAKE_CXXFLAGS += -Werror=sizeof-array-argument
	QMAKE_CXXFLAGS += -Werror=bool-compare
	QMAKE_CXXFLAGS += -Werror=switch-bool
	QMAKE_CXXFLAGS += -Werror=sizeof-array-argument
	QMAKE_CXXFLAGS += -Werror=shadow

	greaterThan(QT_MAJOR_VERSION,4){
		QMAKE_CXXFLAGS += -Werror=suggest-override # Too many warnings in Qt4
#		QMAKE_CXXFLAGS += -Werror=ctor-dtor-privacy # Too many warnings in Qt
	}
}

greaterThan(COMPILER_MAJOR_VERSION, 5){
	# GCC 6+
	QMAKE_CXXFLAGS += -Werror=misleading-indentation
	contains(DEFINES, DEBUG){  # false positive with QPointer in release
		QMAKE_CXXFLAGS += -Werror=null-dereference
	}

	QMAKE_CXXFLAGS += -Werror=duplicated-cond
	QMAKE_CXXFLAGS += -Werror=conversion-null
}
#greaterThan(COMPILER_MAJOR_VERSION, 6){}

