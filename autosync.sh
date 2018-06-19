#!/bin/bash
set -Ceu

DEST='reedbush:/lustre/gj29/j29006/object-recognition/'
EXCLUDE=".git/ MNIST_data/ OUXT_imageData/.git/ OUXT_imageData/images/"

EXCLUDECMD=''
for file in $EXCLUDE
do
  EXCLUDECMD+="--exclude '$file' "
done

CMD="rsync -av -e ssh . $DEST --update --exclude 'autosync.sh' $EXCLUDECMD \
  && rsync -av -e ssh $DEST . --update --exclude 'autosync.sh' $EXCLUDECMD"

watchman-make -p '**/*' --run "$CMD"
