#!/bin/bash

for name in `ls main/ | sed 's/.cpp$//g'`
do
  ./scripts/build_example.sh $name
done
