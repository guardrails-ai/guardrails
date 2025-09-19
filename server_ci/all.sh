#!/bin/bash

tag=${1:-fastapi}

bash ./server_ci/build.sh $tag;
bash ./server_ci/run.sh $tag;
bash ./server_ci/check.sh;
bash ./server_ci/test.sh;
bash ./server_ci/clean.sh;