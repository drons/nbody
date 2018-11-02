include( pri/nbody.pri )

TEMPLATE = subdirs

SUBDIRS += nbody

SUBDIRS += player
SUBDIRS += test

player.subdir = player
player.depends = nbody

test.subdir = test
test.depends = nbody

OTHER_FILES += appveyor.yml .travis.yml README.md .astylerc
