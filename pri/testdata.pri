{
	testdata.name = Copy test data
	testdata.input = TEST_DATA
	testdata.CONFIG += no_link target_predeps

	testdata.commands = $$QMAKE_COPY ${QMAKE_FILE_IN} ${QMAKE_FILE_OUT}
	testdata.output = $$DESTDIR/../data/${QMAKE_FILE_BASE}${QMAKE_FILE_EXT}

	QMAKE_EXTRA_COMPILERS += testdata
}
