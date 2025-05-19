// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "driver_types.h"
#include "paddle/extension.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

struct RemoteCacheKvIpc {
    struct save_cache_kv_complete_signal_layerwise_meta_data{
        int32_t layer_id=-1;
        void * shm_ptr=nullptr;
        int shm_fd=-1;
        save_cache_kv_complete_signal_layerwise_meta_data(){}
        save_cache_kv_complete_signal_layerwise_meta_data(int32_t layer_id_,
                                                            void* shm_ptr_,
                                                            int shm_fd_)
            :layer_id(layer_id_), shm_ptr(shm_ptr_), shm_fd(shm_fd_){
        }
    };
    static RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data kv_complete_signal_meta_data;
    static void* kv_complete_signal_identity_ptr;
    static bool kv_complete_signal_shmem_opened;

    static RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_meta_data open_shm_and_get_complete_signal_meta_data(
        const int rank_id,
        const bool keep_pd_step_flag);
    static void CUDART_CB save_cache_kv_complete_signal_layerwise(void* meta_data);
};
