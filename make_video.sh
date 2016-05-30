#!/bin/bash
mencoder "mf://./video/*.png" -oac copy -ovc lavc -lavcopts vcodec=msmpeg4:vbitrate=8000 -ffourcc MP43 -o ./test.avi

