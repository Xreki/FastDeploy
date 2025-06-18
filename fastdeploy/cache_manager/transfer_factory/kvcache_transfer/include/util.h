#ifndef KVCACHE_UTILS_H
#define KVCACHE_UTILS_H

#include <chrono>
#include <ctime>
#include <iostream>

#define PATH_MAX 4096 /* # chars in a path name including nul */
#define RDMA_WR_LIST_MAX_SIZE 32
#define RDMA_SQ_MAX_SIZE 1024

#define RDMA_DEFAULT_PORT 20001
#define RDMA_TCP_CONNECT_SIZE 1024
#define RDMA_POLL_CQE_TIMEOUT 30

/// @brief Connection status enumeration
enum class ConnStatus {
  kConnected,        // Connection is active
  kDisconnected,     // Connection is not active
  kError,            // Connection error occurred
  kTimeout,          // Connection timed out
  kInvalidParameters // Invalid connection parameters
};

/// @brief Queue Pair (QP) setup result status
enum class QpStatus {
  kSuccess,           // Successfully transitioned QP to RTS
  kInvalidParameters, // ctx or dest is null
  kDeviceQueryFailed, // ibv_query_device failed
  kPortQueryFailed,   // ibv_query_port failed
  kMtuMismatch,       // Requested MTU exceeds active MTU
  kModifyToRTRFailed, // Failed to modify QP to RTR
  kModifyToRTSFailed  // Failed to modify QP to RTS
};

#endif
