#!/bin/bash
./gitize_externals.sh
for ext in `ls`; do
    if [ -f $ext/.git.bak ]; then
        echo "Updating $ext"
        cd $ext
        git pull
        cd ..
    fi
done
./degitize_externals.sh
