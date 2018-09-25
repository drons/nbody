include( pri/nbody.pri )

TEMPLATE = subdirs

SUBDIRS += nbody
SUBDIRS += player
SUBDIRS += test

src.subdir = src
#src.depends =

player.subdir = player
player.depends = nbody

test.subdir = test
test.depends = nbody

OTHER_FILES += .travis.yml README.md
