#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <unistd.h>
#include "paddle/extension.h"

#define MAX_BSZ 512

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ + 2];  // stop_flag, bsz, tokens
};

struct msgdatakv {
    long mtype;
    int mtext[MAX_BSZ * 2 + 2];  // encoder_count, layer_id, bid- pair
};