#!/bin/bash
for ext in `ls`; do
    if [ -f $ext/.git.bak ]; then
        cd $ext
        tar zxf .git.bak
        cd ..
    fi
done
