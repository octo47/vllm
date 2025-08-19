"""Unified block allocator for MLX platform with unified memory architecture."""

import time
import logging
from typing import Dict, FrozenSet, List, Optional, Tuple

from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator
from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator
from vllm.utils import Device

logger = logging.getLogger(__name__)

class UnifiedBlockAllocator(DeviceAwareBlockAllocator):
    """
    Unified block allocator for MLX platform with unified memory architecture.
    
    This allocator manages a single memory pool instead of separate CPU/GPU pools,
    which is suitable for unified memory systems like Apple Silicon where CPU and
    GPU share the same physical memory space.
    
    Key characteristics:
    - Single memory pool: All blocks are managed in one unified pool
    - No-op swapping: swap operations are implemented as no-ops since there's 
      no distinction between CPU and GPU memory
    - Device abstraction: All blocks are treated as GPU blocks for compatibility
      with existing vLLM code while using unified memory underneath
    """

    @staticmethod
    def create(
        allocator_type: str,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        block_size: int,
        device_config=None,  # Unused in unified memory
    ) -> "UnifiedBlockAllocator":
        """Create a unified block allocator.
        
        Args:
            allocator_type: Type of block allocator ("naive" or "prefix_caching")
            num_gpu_blocks: Number of GPU blocks (used as total blocks)
            num_cpu_blocks: Number of CPU blocks (ignored in unified memory)
            block_size: Size of each block in tokens
            device_config: Device configuration (unused)
            
        Returns:
            UnifiedBlockAllocator instance
        """
        # In unified memory, we only use the GPU block count since all memory
        # is accessible by both CPU and GPU components
        total_blocks = num_gpu_blocks + num_cpu_blocks
        
        # Create block IDs for the unified pool
        block_ids = list(range(total_blocks))
        
        # Create the underlying block allocator
        if allocator_type == "naive":
            allocator = NaiveBlockAllocator(
                create_block=NaiveBlock,  # type: ignore
                num_blocks=total_blocks,
                block_size=block_size,
                block_ids=block_ids,
            )
        elif allocator_type == "prefix_caching":
            allocator = PrefixCachingBlockAllocator(
                num_blocks=total_blocks,
                block_size=block_size,
                block_ids=block_ids,
            )
        else:
            raise ValueError(f"Unknown allocator type: {allocator_type}")
        
        logger.info(f"Allocating {__name__}: allocator_type={allocator_type} total_blocks={total_blocks} block_size={block_size}")

        return UnifiedBlockAllocator(
            block_allocator=allocator,
            num_total_blocks=total_blocks,
        )

    def __init__(
        self,
        block_allocator,
        num_total_blocks: int,
    ):
        """Initialize the unified block allocator.
        
        Args:
            block_allocator: The underlying block allocator (naive or prefix caching)
            num_total_blocks: Total number of blocks in the unified pool
        """
        self._allocator = block_allocator
        self._num_total_blocks = num_total_blocks
        
        # Null block for sliding window support
        self._null_block: Optional[Block] = None

    def allocate_mutable_block(
        self,
        prev_block: Optional[Block],
        device: Device,
        extra_hash: Optional[int] = None,
    ) -> Block:
        """Allocate a mutable block.
        
        In unified memory, device parameter is ignored since all memory
        is accessible by both CPU and GPU.
        
        Args:
            prev_block: Previous block in the sequence
            device: Target device (ignored in unified memory)
            extra_hash: Extra hash for prefix caching
            
        Returns:
            Allocated mutable block
        """
        return self._allocator.allocate_mutable_block(
            prev_block=prev_block,
            extra_hash=extra_hash,
        )

    def allocate_immutable_block(
        self,
        prev_block: Optional[Block],
        token_ids: List[int],
        device: Device,
        extra_hash: Optional[int] = None,
    ) -> Block:
        """Allocate an immutable block with specified token IDs.
        
        Args:
            prev_block: Previous block in the sequence
            token_ids: Token IDs to store in the block
            device: Target device (ignored in unified memory)
            extra_hash: Extra hash for prefix caching
            
        Returns:
            Allocated immutable block
        """
        return self._allocator.allocate_immutable_block(
            prev_block=prev_block,
            token_ids=token_ids,
            extra_hash=extra_hash,
        )

    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        device: Device,
        extra_hash: Optional[int] = None,
    ) -> List[Block]:
        """Allocate multiple immutable blocks.
        
        Args:
            prev_block: Previous block in the sequence
            block_token_ids: List of token ID lists for each block
            device: Target device (ignored in unified memory)
            extra_hash: Extra hash for prefix caching
            
        Returns:
            List of allocated immutable blocks
        """
        return self._allocator.allocate_immutable_blocks(
            prev_block=prev_block,
            block_token_ids=block_token_ids,
            extra_hash=extra_hash,
        )

    def get_num_free_blocks(self, device: Device) -> int:
        """Get number of free blocks.
        
        Args:
            device: Device type (ignored in unified memory)
            
        Returns:
            Number of free blocks in the unified pool
        """
        return self._allocator.get_num_free_blocks()

    def get_num_total_blocks(self, device: Device) -> int:
        """Get total number of blocks.
        
        Args:
            device: Device type (ignored in unified memory)
            
        Returns:
            Total number of blocks in the unified pool
        """
        return self._num_total_blocks

    def free(self, block: Block) -> None:
        """Free a block and return it to the unified pool.
        
        Args:
            block: Block to free
        """
        self._allocator.free(block)

    def fork(self, last_block: Block) -> List[Block]:
        """Fork a sequence by creating a copy of the block chain.
        
        Args:
            last_block: Last block in the sequence to fork
            
        Returns:
            List of blocks in the forked sequence
        """
        return self._allocator.fork(last_block)

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        """Get all block IDs managed by this allocator.
        
        Returns:
            Frozen set of all block IDs
        """
        return self._allocator.all_block_ids

    def clear_copy_on_writes(self) -> List[Tuple[int, int]]:
        """Clear copy-on-write references.
        
        Returns:
            List of copy-on-write mappings that were cleared
        """
        return self._allocator.clear_copy_on_writes()

    def mark_blocks_as_accessed(self, block_ids: List[int], now: float) -> None:
        """Mark blocks as recently accessed for prefix caching.
        
        Args:
            block_ids: List of block IDs to mark as accessed
            now: Current timestamp
        """
        return self._allocator.mark_blocks_as_accessed(block_ids, now)

    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as computed for prefix caching.
        
        Args:
            block_ids: List of block IDs to mark as computed
        """
        return self._allocator.mark_blocks_as_computed(block_ids)

    def get_common_computed_block_ids(
        self, computed_seq_block_ids: List[List[int]]
    ) -> List[int]:
        """Get common computed block IDs across sequences.
        
        Args:
            computed_seq_block_ids: List of computed block ID lists for each sequence
            
        Returns:
            List of common computed block IDs
        """
        return self._allocator.get_common_computed_block_ids(computed_seq_block_ids)

    def get_num_full_blocks_touched(self, blocks: List[Block], device: Device) -> int:
        """Get number of full blocks that would be touched by operations.
        
        Args:
            blocks: List of blocks to analyze
            device: Target device (ignored in unified memory)
            
        Returns:
            Number of full blocks that would be touched
        """
        return self._allocator.get_num_full_blocks_touched(blocks)

    def swap(
        self, blocks: List[Block], src_device: Device, dst_device: Device
    ) -> Dict[int, int]:
        """Swap operation - implemented as no-op in unified memory.
        
        In unified memory systems like MLX, there is no distinction between
        CPU and GPU memory, so swap operations are not needed. This method
        returns an empty mapping to indicate no swapping occurred.
        
        Args:
            blocks: Blocks to swap (ignored)
            src_device: Source device (ignored) 
            dst_device: Destination device (ignored)
            
        Returns:
            Empty dictionary since no swapping is performed
        """
        # No-op: unified memory doesn't require swapping between devices
        return {}

    def get_physical_block_id(self, device: Device, absolute_id: int) -> int:
        """Get physical block ID for a given absolute ID.
        
        Args:
            device: Device type (ignored in unified memory)
            absolute_id: Absolute block ID
            
        Returns:
            Physical block ID (same as absolute ID in unified memory)
        """
        return self._allocator.get_physical_block_id(absolute_id)

    def allocate_or_get_null_block(self) -> Block:
        """Get or create a null block for sliding window support.
        
        Null blocks are used as placeholders for KV cache blocks that have
        been dropped due to sliding window.
        
        Returns:
            Null block instance
        """
        if self._null_block is None:
            # Create a null block using a regular mutable block
            # We use Device.GPU since all blocks are treated as GPU blocks
            # in unified memory for compatibility
            regular_block = self.allocate_mutable_block(
                prev_block=None,
                device=Device.GPU
            )
            self._null_block = NullBlock(regular_block)
        return self._null_block

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Get prefix cache hit rate.
        
        Args:
            device: Device type (ignored in unified memory)
            
        Returns:
            Prefix cache hit rate, -1 if not supported
        """
        return self._allocator.get_prefix_cache_hit_rate()

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        """Reset prefix cache.
        
        Args:
            device: Device type (ignored in unified memory)
            
        Returns:
            True if reset was successful
        """
        return self._allocator.reset_prefix_cache()

    def find_cached_blocks_prefix(
        self, block_hashes: List[int], device: Device = Device.GPU
    ) -> List[int]:
        """Find cached blocks with matching hash prefix.
        
        Args:
            block_hashes: List of block hashes to search for
            device: Device type (ignored in unified memory)
            
        Returns:
            List of matching cached block IDs
        """
        return self._allocator.find_cached_blocks_prefix(block_hashes)


class NullBlock(Block):
    """
    Null blocks are used as placeholders for KV cache blocks that have
    been dropped due to sliding window.
    
    This implementation wraps an ordinary block and prevents it from
    being modified while allowing testing via isinstance().
    """

    def __init__(self, proxy: Block):
        super().__init__()
        self._proxy = proxy

    def append_token_ids(self, token_ids: List[int]) -> None:
        raise ValueError("null block should not be modified")

    @property
    def block_id(self) -> Optional[int]:
        return self._proxy.block_id

    @block_id.setter
    def block_id(self, value: Optional[int]) -> None:
        raise ValueError("null block should not be modified")

    @property
    def token_ids(self) -> List[int]:
        return self._proxy.token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError("num_tokens_total is not used for null block")

    @property
    def num_empty_slots(self) -> int:
        return self._proxy.num_empty_slots

    @property
    def is_full(self) -> bool:
        return self._proxy.is_full

    @property
    def prev_block(self) -> Optional[Block]:
        return self._proxy.prev_block

    @property
    def extra_hash(self) -> Optional[int]:
        return None

    @property
    def computed(self) -> bool:
        return self._proxy.computed

    @computed.setter
    def computed(self, value: bool) -> None:
        self._proxy.computed = value

    @property
    def last_accessed(self) -> float:
        return self._proxy.last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float) -> None:
        self._proxy.last_accessed = last_accessed_ts

    @property
    def content_hash(self) -> Optional[int]:
        return self._proxy.content_hash