"""
进程间通信结构体
"""

import os
import threading
import socket
import time
import numpy as np
from multiprocessing.managers import (AcquirerProxy, BaseManager, ListProxy,
                                      Value, ValueProxy)
from queue import Queue
from fastdeployllm.utils import llm_logger
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Dict, Tuple, List, Any


def shared_memory_exists(name: str) -> bool:
    """Check if a shared memory block with the given name exists.

    Args:
        name: The unique identifier of the shared memory block.

    Returns:
        True if the shared memory exists, False otherwise.
    """
    try:
        shm = SharedMemory(name=name, create=False)
        shm.close()
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


class IPCSignal:
    """A shared memory wrapper for inter-process communication using numpy arrays.

    Allows creating or connecting to existing shared memory blocks and synchronizing
    numpy array data between processes.

    Attributes:
        shm: The underlying SharedMemory object.
        value: Numpy array interface to the shared memory buffer.
    """

    def __init__(self,
                 name: str,
                 array: np.ndarray,
                 dtype: np.dtype,
                 create: bool = True) -> None:
        """Initialize or connect to a shared memory block.

        Args:
            name: Unique identifier for the shared memory block.
            array: Numpy array template defining shape and data type.
            dtype: Data type of the array (must match array.dtype).
            create: If True, creates new memory block; otherwise connects to existing.

        Raises:
            AssertionError: If create=True but memory already exists, or dtype mismatch.
        """
        assert isinstance(array, np.ndarray), "Input must be a numpy array"
        assert dtype == array.dtype, "Specified dtype must match array dtype"

        if create:
            assert not shared_memory_exists(
                name), f"ShareMemory: {name} already exists"
            self.shm = SharedMemory(create=True, size=array.nbytes, name=name)
            self.value: np.ndarray = np.ndarray(array.shape,
                                                dtype=array.dtype,
                                                buffer=self.shm.buf)
            self.value[:] = array  # Initialize with input array data
        else:
            self.shm = SharedMemory(name=name)
            self.value: np.ndarray = np.ndarray(array.shape,
                                                dtype=array.dtype,
                                                buffer=self.shm.buf)

    def clear(self) -> None:
        """Release system resources and unlink the shared memory block."""
        self.shm.close()
        self.shm.unlink()


class EngineWorkerQueue:
    """
    Cross-machine and cross-process communication queue between Engine and Worker.
    Manages shared resources using multiprocessing managers for inter-process communication.
    """

    def __init__(self,
                 address: Tuple[str, int] = ('0.0.0.0', 5000),
                 authkey: bytes = b'secret_key',
                 is_server: bool = False,
                 num_client: int = 1,
                 client_id: int = -1) -> None:
        """
        Initialize the communication queue.

        Args:
            address: Network address (IP, port) for the queue server
            authkey: Authentication key for secure connection
            is_server: Whether this instance acts as a server
            num_client: Total number of expected clients
            client_id: Unique identifier for client instances
        """
        self.address: Tuple[str, int] = address
        self.authkey: bytes = authkey
        self.num_client: int = num_client
        self.client_id: int = client_id

        # Custom QueueManager for proxy object registration
        class QueueManager(BaseManager):
            pass

        if is_server:
            # Server-side initialization for shared resources
            self.tasks_init: List[Any] = list()
            self.client_read_flag_init: List[int] = [1] * self.num_client
            self.lock_init: threading.Lock = threading.Lock()
            self.read_finish_flag_init: Value = Value("i", 0)
            self.connected_client_counter_init: Value = Value("i", 0)

            # Register shared objects with proxy types
            QueueManager.register("get_tasks",
                                  callable=lambda: self.tasks_init,
                                  proxytype=ListProxy)
            QueueManager.register("get_client_read_flag",
                                  callable=lambda: self.client_read_flag_init,
                                  proxytype=ListProxy)
            QueueManager.register("get_lock",
                                  callable=lambda: self.lock_init,
                                  proxytype=AcquirerProxy)
            QueueManager.register("get_read_finish_flag",
                                  callable=lambda: self.read_finish_flag_init,
                                  proxytype=ValueProxy)
            QueueManager.register(
                "get_connected_client_counter",
                callable=lambda: self.connected_client_counter_init,
                proxytype=ValueProxy)

            self.manager: BaseManager = QueueManager(address=self.address,
                                                     authkey=self.authkey)
            self.manager.start()
        else:
            # Client-side connection setup
            assert self.client_id >= 0 and self.client_id < self.num_client, (
                f"self.client_id={self.client_id}, self.num_client={self.num_client}"
            )
            QueueManager.register("get_tasks")
            QueueManager.register("get_client_read_flag")
            QueueManager.register("get_lock")
            QueueManager.register("get_read_finish_flag")
            QueueManager.register("get_connected_client_counter")
            self.manager = QueueManager(address=self.address,
                                        authkey=self.authkey)
            self._connect_with_retry()

        # Get proxy objects for shared resources
        self.tasks: ListProxy = self.manager.get_tasks()
        self.client_read_flag: ListProxy = self.manager.get_client_read_flag()
        self.lock: AcquirerProxy = self.manager.get_lock()
        self.read_finish_flag: ValueProxy = self.manager.get_read_finish_flag()
        self.connected_client_counter: ValueProxy = self.manager.get_connected_client_counter(
        )
        assert self.num_client == len(self.client_read_flag)

        if is_server:
            llm_logger.info(f"EngineWorkerQueue server started.")
        else:
            # Update client connection counter
            self.lock.acquire()
            self.connected_client_counter.set(
                self.connected_client_counter.get() + 1)
            self.lock.release()
            llm_logger.info((
                f"Connected EngineWorkerQueue client_id: {self.client_id}, number "
                f"of connected clients: {self.connected_client_counter.get()}"
            ))

    def _connect_with_retry(self,
                            max_retries: int = 5,
                            interval: int = 3) -> None:
        """
        Connect to the server with retry mechanism.

        Args:
            max_retries: Maximum connection attempts
            interval: Retry interval in seconds

        Raises:
            ConnectionError: If all connection attempts fail
        """
        for _ in range(max_retries):
            try:
                self.manager.connect()
                return
            except ConnectionRefusedError:
                time.sleep(interval)
        raise ConnectionError(f"TaskQueue cannot connect {self.address}")

    def put_tasks(self, tasks: List[Any]) -> None:
        """
        Add tasks to the shared queue in a thread-safe manner.
        Waits until all clients have read previous tasks before adding new ones.

        Args:
            tasks: Tasks to be added to the queue
        """
        self.lock.acquire()
        while sum(self.client_read_flag) < self.num_client:
            self.lock.release()
            time.sleep(0.001)
            self.lock.acquire()

        self.tasks[:] = list()
        self.client_read_flag[:] = [0] * self.num_client
        self.tasks.append(tasks)
        self.lock.release()

    def get_tasks(self) -> Tuple[List[Any], bool]:
        """
        Retrieve tasks from the shared queue and update read status.

        Returns:
            tuple: (list of tasks, bool indicating if all clients have read)
        """
        tasks: List[Any] = list()
        self.lock.acquire()
        tasks.extend(self.tasks)
        self.client_read_flag[self.client_id] = 1
        all_client_read: bool = np.sum(
            self.client_read_flag) == self.num_client
        if all_client_read:
            self.tasks[:] = list()
        self.lock.release()
        return tasks, all_client_read

    def num_tasks(self) -> int:
        """
        Get current number of tasks in the queue.

        Returns:
            int: Total number of tasks
        """
        self.lock.acquire()
        total_num: int = len(self.tasks)
        self.lock.release()
        return total_num
