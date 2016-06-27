#!/bin/bash
./gitize_externals.sh
for ext in `ls`; do
    if [ -f $ext/.git.bak ]; then
        echo "Checking $ext"
        cd $ext
        git fetch
        git status
        cd ..
    fi
done
./degitize_externals.sh
