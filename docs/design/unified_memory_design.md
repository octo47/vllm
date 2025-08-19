# Unified Memory Architecture for vLLM

This document outlines the technical design for a unified memory architecture in vLLM, intended to support devices with unified memory systems, such as Apple Silicon.

## 1. High-Level Design

The proposed unified memory architecture will replace the existing dual-system (CPU/GPU) memory management with a single, unified memory pool. This will eliminate the need for explicit data swapping between CPU and GPU, allowing the operating system to manage memory paging more efficiently. This design is inspired by PyTorch's MPS backend and aims to simplify the memory management logic while improving performance on unified memory devices.

The core of this design is a new `UnifiedBlockAllocator` that will manage a single pool of memory blocks. The existing `Scheduler` and `BlockSpaceManager` will interact with this new allocator through the `DeviceAwareBlockAllocator` interface, ensuring that the core scheduling logic remains unchanged.

## 2. Component Modifications

### 2.1. KV Cache Management

The `CacheEngine` will be modified to use the new `UnifiedBlockAllocator` when running on a unified memory platform. The `swap_in` and `swap_out` methods in the `CacheEngine` will become no-ops, as the operating system will handle memory management.

### 2.2. Model Weight and Tensor Allocation

Model weights and other tensors will be allocated in the unified memory pool. The `MlxPlatform` will be updated to report the total system memory as available "device" memory, allowing vLLM to use the full memory capacity of the system.

### 2.3. Scheduler

No significant changes are required for the `Scheduler`. It will continue to issue `swap` commands, but these will be handled by the `UnifiedBlockAllocator`, which will implement them as no-ops.

## 3. New Abstractions

### 3.1. `UnifiedBlockAllocator`

This new class will be responsible for managing the unified memory pool. It will implement the `DeviceAwareBlockAllocator` interface and will have the following key characteristics:

-   **Single Memory Pool:** It will manage a single pool of memory blocks, rather than separate pools for CPU and GPU.
-   **No-Op Swapping:** The `swap_in` and `swap_out` methods will be implemented as no-ops, as memory management will be handled by the operating system.
-   **Platform-Specific Instantiation:** The `MlxPlatform` will be responsible for creating an instance of the `UnifiedBlockAllocator`.

## 4. Configuration and Integration

The unified memory architecture will be enabled automatically when vLLM is run on a supported platform, such as Apple Silicon with the MLX backend. No special configuration or command-line arguments will be required. The `MlxPlatform` will detect the unified memory architecture and instantiate the `UnifiedBlockAllocator`.

## 5. Backwards Compatibility

The new architecture will be fully backward compatible with existing device types. The `Platform` abstraction will ensure that the correct block allocator is used for each platform:

-   **CUDA/ROCm:** The existing `CpuGpuBlockAllocator` will be used.
-   **MLX (Apple Silicon):** The new `UnifiedBlockAllocator` will be used.

This approach ensures that the unified memory architecture can be integrated seamlessly without affecting existing functionality.