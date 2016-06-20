#!/bin/bash
for ext in `ls`; do
    if [ -f $ext/.git.bak ]; then
        cd $ext
        tar zcf .git.bak .git
        rm -rf .git
        cd ..
    fi
done
