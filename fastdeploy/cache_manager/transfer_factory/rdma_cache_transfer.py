"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import os
import rdma_comm

from fastdeploy.utils import get_logger

logger = get_logger(f"cache_messager", f"cache_messager.log")

class RDMACommManager:
    """
    RDMACommManager to manage rdma communication
    """

    def __init__(self, splitwise_role, rank, gpu_id, cache_k_ptr_list, \
                cache_v_ptr_list, max_block_num, block_bytes, rdma_port):
        self.messager = rdma_comm.RDMACommunicator(splitwise_role,
                                    rank,
                                    str(rdma_port) if splitwise_role == "decode" else "0",
                                    cache_k_ptr_list,
                                    cache_v_ptr_list,
                                    max_block_num,
                                    block_bytes,
                                )
        self.splitwise_role = splitwise_role
        self.connected_rdma = set()
        rdma_ips = self.get_rdma_ip()
        logger.info(f"local rdma ip: {rdma_ips}")
        self.save_decode_rdma_ip(f"/dev/tmp/rdma_ips_{gpu_id}.txt", rdma_ips)
        logger.info(f"init rdma messager {gpu_id} {rdma_ips} {rdma_port}")

    
    def connect(self, ip, port):
        """prefill连接decocde"""
        assert self.splitwise_role == "prefill", "only prefill can call this method"
        addr = f"{ip}:{str(port)}"
        if addr in self.connected_rdma:
            return True
        ret = self.messager.is_connected(ip, str(port))
        if ret:
            self.connected_rdma.add(addr)
            return True

        ret = self.messager.connect(ip, str(port))
        logger.info(f"connect to remote rdma address {ip}:{port} status is {ret}")
        if ret == 0:
            self.connected_rdma.add(addr)
        return ret == 0
    

    def get_rdma_ip(self):
        """ get rdma ip """
        return self.messager.fetch_local_ip()

    def save_decode_rdma_ip(self, file_path, ip):
        """写入 IP 地址到文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(ip + "\n")

    def write_cache(self, ip, port, local_block_ids, remote_block_ids, layer_idx):
        """
        Connect to remote gpu and write cache.
        """
        return self.messager.write_cache(ip, str(port), local_block_ids, remote_block_ids, layer_idx)
        
    def is_write_completed(self, target_ip, target_id, src_block_ids) :
        """
        Check whether all blocks are written successfully.
        """

        return self.messager.is_write_completed(target_ip, str(target_id), 2 * len(src_block_ids))
