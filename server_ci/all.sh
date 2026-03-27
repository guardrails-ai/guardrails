#!/bin/bash

bash ./server_ci/build.sh;
bash ./server_ci/run.sh;
bash ./server_ci/check.sh;
bash ./server_ci/test.sh;
bash ./server_ci/clean.sh;