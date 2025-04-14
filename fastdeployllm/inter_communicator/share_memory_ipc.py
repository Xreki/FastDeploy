"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import numpy as np
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Dict, Tuple

class SharedMemoryIPC:
    _registry: Dict[str, Tuple[SharedMemory, np.ndarray, Optional[multiprocessing.Lock]]] = {}

    @classmethod
    def add_shared_memory(
        cls,
        key: str,
        shape: tuple,
        dtype: np.dtype,
        create: bool = True,
        with_lock: bool = True
    ) -> np.ndarray:
        """
        Registers a shared memory block and binds a numpy array view

        Args:
            key: Unique identifier for the memory block
            shape: Shape of the numpy array
            dtype: Numpy data type for the array elements
            create: Whether to create new memory block (True) or attach to existing (False)
            with_lock: Whether to associate a mutex lock for synchronization

        Returns:
            Numpy array view that directly operates on shared memory
        """
        # Calculate required memory size
        item_size = np.dtype(dtype).itemsize
        size = int(np.prod(shape)) * item_size

        try:
            shm = SharedMemory(name=key, create=create, size=size)
        except FileExistsError:
            shm = SharedMemory(name=key)

        # Create numpy array view
        arr = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=shm.buf,
            order='C'  # Ensure contiguous memory layout
        )

        # Initialize memory (only when creating new block)
        if create:
            arr.fill(0)

        lock = multiprocessing.Lock() if with_lock else None
        cls._registry[key] = (shm, arr, lock)
        return arr

    @classmethod
    def write(
        cls,
        key: str,
        data: np.ndarray,
        use_lock: Optional[bool] = None
    ) -> None:
        """
        Writes numpy data to shared memory

        Args:
            data: Numpy array to write (must match registered shape and dtype)
        """
        shm, arr, lock = cls._get_memory_block(key)

        # Validate data compatibility
        if data.shape != arr.shape:
            raise ValueError(f"Shape mismatch: registered {arr.shape} vs input {data.shape}")
        if data.dtype != arr.dtype:
            raise TypeError(f"Dtype mismatch: registered {arr.dtype} vs input {data.dtype}")

        # Perform write operation
        def _write():
            np.copyto(arr, data)  # Memory-level efficient copy

        cls._execute_with_lock(lock, use_lock, _write)

    @classmethod
    def read(
        cls,
        key: str,
        use_lock: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns a read-only view of shared memory (avoids memory copying)
        """
        shm, arr, lock = cls._get_memory_block(key)

        # Create read-only view
        def _read():
            return arr.view()  # Return view of original array

        return cls._execute_with_lock(lock, use_lock, _read)

    @classmethod
    def _execute_with_lock(cls, lock, use_lock, func):
        """Executes function with optional locking"""
        if use_lock is None:
            use_lock = (lock is not None)

        if use_lock and lock is None:
            raise RuntimeError("Lock required but not available")

        if use_lock:
            with lock:
                return func()
        else:
            return func()

    @classmethod
    def _get_memory_block(cls, key: str):
        """Retrieves registered memory block"""
        if key not in cls._registry:
            raise KeyError(f"Shared memory block '{key}' not found")
        return cls._registry[key]

    @classmethod
    def cleanup(cls, key: str) -> None:
        """Releases shared memory resources"""
        if key in cls._registry:
            shm, arr, _ = cls._registry.pop(key)
            arr.base = None  # Disassociate buffer
            shm.close()
            try:
                shm.unlink()  # Completely remove shared memory block
            except FileNotFoundError:
                pass  # Already unlinked by another process
