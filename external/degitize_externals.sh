#!/bin/bash
for ext in `ls`; do
    if [ -d $ext/.git ]; then
        cd $ext
        tar zcf .git.bak .git
        rm -rf .git
        cd ..
    fi
done
