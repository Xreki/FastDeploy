#!/bin/bash

if [ "$MODEL" = "eb45t" ]; then
    bash ./fastdeploy/agent/prepare_eff.sh
else
    bash ./fastdeploy/agent/prepare_no_eff.sh
fi
