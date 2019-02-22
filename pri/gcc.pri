DEFINES += HAVE_OPENMP

QMAKE_CXXFLAGS += -std=gnu++11
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
QMAKE_CXXFLAGS += -pedantic-errors

release:equals(QT_MAJOR_VERSION,4){
	QMAKE_CXXFLAGS += -Wno-unused-variable
}

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
	QMAKE_CXXFLAGS += -Wno-ignored-attributes
}

greaterThan(COMPILER_MAJOR_VERSION, 6){
	# GCC 7+
	QMAKE_CXXFLAGS += -Werror=pointer-compare
	#QMAKE_CXXFLAGS += -Werror=duplicated-branches #Errors in opencv2
	QMAKE_CXXFLAGS += -Werror=switch-unreachable
	QMAKE_CXXFLAGS += -Werror=restrict
	QMAKE_CXXFLAGS += -Werror=shadow=local
	QMAKE_CXXFLAGS += -Werror=nonnull
}

greaterThan(COMPILER_MAJOR_VERSION, 7){
	# GCC 8+
	QMAKE_CXXFLAGS += -Werror=multistatement-macros
	QMAKE_CXXFLAGS += -Werror=stringop-truncation
	QMAKE_CXXFLAGS += -Werror=cast-align=strict
	QMAKE_CXXFLAGS += -Werror=old-style-cast
}

contains( CONFIG, build-gcov ){
	QMAKE_CFLAGS += -fprofile-arcs -ftest-coverage
	QMAKE_CXXFLAGS += -fprofile-arcs -ftest-coverage
	QMAKE_LFLAGS += -lgcov --coverage
}

contains( CONFIG, build-asan ){
	#-fcheck-pointer-bounds
	greaterThan(COMPILER_MAJOR_VERSION, 4){
		ASAN_FLAGS += -fsanitize=float-divide-by-zero
		ASAN_FLAGS += -fsanitize=float-cast-overflow
		ASAN_FLAGS += -fsanitize=bounds
		ASAN_FLAGS += -fsanitize=alignment
		ASAN_FLAGS += -fsanitize=object-size
		ASAN_FLAGS += -fsanitize=vptr
#		ASAN_FLAGS += -fsanitize=thread
		ASAN_FLAGS += -fsanitize=address
	}
	greaterThan(COMPILER_MAJOR_VERSION, 6){
		ASAN_FLAGS += -fsanitize=bounds-strict
	}
	greaterThan(COMPILER_MAJOR_VERSION, 6){
		ASAN_FLAGS += -fsanitize=pointer-compare
		ASAN_FLAGS += -fsanitize=pointer-subtract
		ASAN_FLAGS += -fsanitize=undefined
		ASAN_FLAGS += -fsanitize=builtin
		ASAN_FLAGS += -fsanitize=pointer-overflow
	}

	QMAKE_CXXFLAGS += $$ASAN_FLAGS
	QMAKE_LFLAGS += $$ASAN_FLAGS

	# With ASAN we must turnoff OpenMP
	DEFINES -= HAVE_OPEN_MP
	# https://stackoverflow.com/questions/50024731/ld-unrecognized-option-push-state-no-as-needed
	QMAKE_LFLAGS += -fuse-ld=gold
}
