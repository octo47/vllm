# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from typing import Optional, List
import time

from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.unified_block_allocator import UnifiedBlockAllocator, NullBlock
from vllm.utils import Device


class TestUnifiedBlockAllocator:
    """Test suite for UnifiedBlockAllocator class."""

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    @pytest.mark.parametrize("num_gpu_blocks", [1, 16, 1024])
    @pytest.mark.parametrize("num_cpu_blocks", [0, 16, 512])  # Should be ignored
    @pytest.mark.parametrize("block_size", [1, 8, 16])
    def test_create_allocator(allocator_type: str, num_gpu_blocks: int, 
                            num_cpu_blocks: int, block_size: int):
        """Test UnifiedBlockAllocator creation with different configurations."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,  # Should be ignored in unified memory
            block_size=block_size,
        )
        
        # In unified memory, total blocks should equal GPU blocks (CPU blocks ignored)
        assert allocator.get_num_total_blocks(Device.GPU) == num_gpu_blocks
        assert allocator.get_num_total_blocks(Device.CPU) == num_gpu_blocks  # Same for unified memory
        assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks
        assert allocator.get_num_free_blocks(Device.CPU) == num_gpu_blocks  # Same for unified memory

    @staticmethod
    def test_create_allocator_invalid_type():
        """Test that invalid allocator types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown allocator type"):
            UnifiedBlockAllocator.create(
                allocator_type="invalid_type",
                num_gpu_blocks=10,
                num_cpu_blocks=5,
                block_size=8,
            )

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    @pytest.mark.parametrize("num_blocks", [1, 16, 100])
    @pytest.mark.parametrize("block_size", [4, 8, 16])
    def test_allocate_mutable_block(allocator_type: str, num_blocks: int, block_size: int):
        """Test mutable block allocation in unified memory."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=num_blocks,
            num_cpu_blocks=0,
            block_size=block_size,
        )
        
        # Device parameter should be ignored in unified memory
        for device in [Device.GPU, Device.CPU]:
            blocks = []
            for i in range(num_blocks):
                block = allocator.allocate_mutable_block(
                    prev_block=None, 
                    device=device
                )
                blocks.append(block)
                assert block.block_id is not None
                assert allocator.get_num_free_blocks(device) == num_blocks - i - 1
            
            # Should be out of blocks now
            assert allocator.get_num_free_blocks(device) == 0
            
            # Free all blocks
            for block in blocks:
                allocator.free(block)
                assert block.block_id is None
            
            assert allocator.get_num_free_blocks(device) == num_blocks

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    @pytest.mark.parametrize("num_blocks", [4, 16])
    @pytest.mark.parametrize("block_size", [2, 4])
    def test_allocate_immutable_block(allocator_type: str, num_blocks: int, block_size: int):
        """Test immutable block allocation with token IDs."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=num_blocks,
            num_cpu_blocks=0,
            block_size=block_size,
        )
        
        # Create unique token sequences
        token_sequences = [list(range(i * block_size, (i + 1) * block_size)) 
                          for i in range(num_blocks)]
        
        blocks = []
        for i, token_ids in enumerate(token_sequences):
            block = allocator.allocate_immutable_block(
                prev_block=None,
                token_ids=token_ids,
                device=Device.GPU  # Device ignored in unified memory
            )
            blocks.append(block)
            assert block.token_ids == token_ids
            assert allocator.get_num_free_blocks(Device.GPU) == num_blocks - i - 1
        
        # Free all blocks
        for block in blocks:
            allocator.free(block)
        
        assert allocator.get_num_free_blocks(Device.GPU) == num_blocks

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    @pytest.mark.parametrize("num_blocks", [6])
    @pytest.mark.parametrize("block_size", [2])
    def test_allocate_immutable_blocks(allocator_type: str, num_blocks: int, block_size: int):
        """Test multiple immutable block allocation."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=num_blocks,
            num_cpu_blocks=0,
            block_size=block_size,
        )
        
        # Create multiple token sequences
        block_token_ids = [
            [i * block_size + j for j in range(block_size)]
            for i in range(3)  # Allocate 3 blocks at once
        ]
        
        blocks = allocator.allocate_immutable_blocks(
            prev_block=None,
            block_token_ids=block_token_ids,
            device=Device.GPU
        )
        
        assert len(blocks) == 3
        for i, block in enumerate(blocks):
            assert block.token_ids == block_token_ids[i]
        
        assert allocator.get_num_free_blocks(Device.GPU) == num_blocks - 3
        
        # Free blocks
        for block in blocks:
            allocator.free(block)

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    def test_swap_is_noop(allocator_type: str):
        """Test that swap operations are no-ops in unified memory."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=10,
            num_cpu_blocks=5,  # Ignored
            block_size=4,
        )
        
        # Allocate some blocks
        blocks = [
            allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
            for _ in range(3)
        ]
        
        # Swap operation should return empty dict (no-op)
        result = allocator.swap(blocks, Device.GPU, Device.CPU)
        assert result == {}
        
        # Blocks should remain unchanged
        for block in blocks:
            assert block.block_id is not None

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    def test_device_abstraction(allocator_type: str):
        """Test that device parameters are properly abstracted in unified memory."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=20,
            num_cpu_blocks=10,  # Should be ignored
            block_size=8,
        )
        
        # All device queries should return the same result
        assert (allocator.get_num_total_blocks(Device.GPU) == 
                allocator.get_num_total_blocks(Device.CPU) == 20)
        assert (allocator.get_num_free_blocks(Device.GPU) == 
                allocator.get_num_free_blocks(Device.CPU) == 20)
        
        # Allocate a block with GPU device
        gpu_block = allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        
        # Free blocks count should decrease for both "devices"
        assert (allocator.get_num_free_blocks(Device.GPU) == 
                allocator.get_num_free_blocks(Device.CPU) == 19)
        
        # Allocate a block with CPU device (same underlying pool)
        cpu_block = allocator.allocate_mutable_block(prev_block=None, device=Device.CPU)
        
        assert (allocator.get_num_free_blocks(Device.GPU) == 
                allocator.get_num_free_blocks(Device.CPU) == 18)

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    def test_fork_functionality(allocator_type: str):
        """Test block sequence forking."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # Create a chain of blocks
        block1 = allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        block2 = allocator.allocate_mutable_block(prev_block=block1, device=Device.GPU)
        
        # Fork the sequence
        forked_blocks = allocator.fork(block2)
        
        # Should get a list containing the forked sequence
        assert isinstance(forked_blocks, list)
        assert len(forked_blocks) >= 1

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    def test_get_physical_block_id(allocator_type: str):
        """Test physical block ID retrieval."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # In unified memory, physical ID should match absolute ID
        for absolute_id in range(5):
            physical_id = allocator.get_physical_block_id(Device.GPU, absolute_id)
            assert isinstance(physical_id, int)

    @staticmethod
    def test_all_block_ids():
        """Test all_block_ids property."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=5,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        block_ids = allocator.all_block_ids
        assert len(block_ids) == 5
        assert all(isinstance(block_id, int) for block_id in block_ids)

    @staticmethod
    def test_clear_copy_on_writes():
        """Test clearing copy-on-write references."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        cow_mappings = allocator.clear_copy_on_writes()
        assert isinstance(cow_mappings, list)

    @staticmethod
    def test_mark_blocks_operations():
        """Test block marking operations for prefix caching."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="prefix_caching",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # Allocate some blocks first so they can be marked
        blocks = [
            allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
            for _ in range(3)
        ]
        block_ids = [block.block_id for block in blocks]
        now = time.time()
        
        # These should not raise exceptions
        allocator.mark_blocks_as_accessed(block_ids, now)
        allocator.mark_blocks_as_computed(block_ids)

    @staticmethod
    def test_get_common_computed_block_ids():
        """Test getting common computed block IDs."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="prefix_caching",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        computed_seq_block_ids = [[0, 1, 2], [0, 1], [0, 1, 2, 3]]
        common_ids = allocator.get_common_computed_block_ids(computed_seq_block_ids)
        assert isinstance(common_ids, list)

    @staticmethod
    def test_get_num_full_blocks_touched():
        """Test getting number of full blocks touched."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        blocks = [
            allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
            for _ in range(3)
        ]
        
        num_touched = allocator.get_num_full_blocks_touched(blocks, Device.GPU)
        assert isinstance(num_touched, int)
        assert num_touched >= 0

    @staticmethod
    def test_prefix_cache_operations():
        """Test prefix cache related operations."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="prefix_caching",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # Test cache hit rate
        hit_rate = allocator.get_prefix_cache_hit_rate(Device.GPU)
        assert isinstance(hit_rate, float)
        
        # Test cache reset
        reset_result = allocator.reset_prefix_cache(Device.GPU)
        assert isinstance(reset_result, bool)
        
        # Test finding cached blocks
        block_hashes = [12345, 67890, 11111]
        cached_blocks = allocator.find_cached_blocks_prefix(block_hashes, Device.GPU)
        assert isinstance(cached_blocks, list)


class TestNullBlock:
    """Test suite for NullBlock class."""

    @staticmethod
    def test_null_block_creation():
        """Test NullBlock creation and basic properties."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # Get null block
        null_block = allocator.allocate_or_get_null_block()
        assert isinstance(null_block, NullBlock)
        
        # Should return same instance on subsequent calls
        null_block2 = allocator.allocate_or_get_null_block()
        assert null_block is null_block2

    @staticmethod
    def test_null_block_properties():
        """Test NullBlock properties and behavior."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="prefix_caching",  # Use prefix_caching which implements computed
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        null_block = allocator.allocate_or_get_null_block()
        
        # Test readable properties
        assert null_block.block_id is not None
        assert isinstance(null_block.token_ids, list)
        assert isinstance(null_block.num_empty_slots, int)
        assert isinstance(null_block.is_full, bool)
        assert null_block.extra_hash is None
        assert isinstance(null_block.computed, bool)
        assert isinstance(null_block.last_accessed, (int, float))
        
        # Content hash should be accessible
        content_hash = null_block.content_hash
        # Can be None or int

    @staticmethod
    def test_null_block_immutability():
        """Test that NullBlock prevents modification."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        null_block = allocator.allocate_or_get_null_block()
        
        # These operations should raise ValueError
        with pytest.raises(ValueError, match="null block should not be modified"):
            null_block.append_token_ids([1, 2, 3])
        
        with pytest.raises(ValueError, match="null block should not be modified"):
            null_block.block_id = 12345

    @staticmethod
    def test_null_block_not_implemented():
        """Test methods that should raise NotImplementedError."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        null_block = allocator.allocate_or_get_null_block()
        
        with pytest.raises(NotImplementedError, match="num_tokens_total is not used for null block"):
            _ = null_block.num_tokens_total

    @staticmethod
    def test_null_block_mutable_properties():
        """Test that some NullBlock properties can still be set."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="prefix_caching",  # Use prefix_caching which implements computed
            num_gpu_blocks=10,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        null_block = allocator.allocate_or_get_null_block()
        
        # These should work (they modify the proxy, not the null block itself)
        original_computed = null_block.computed
        null_block.computed = not original_computed
        assert null_block.computed == (not original_computed)
        
        original_timestamp = null_block.last_accessed
        new_timestamp = time.time()
        null_block.last_accessed = new_timestamp
        assert null_block.last_accessed == new_timestamp


class TestUnifiedBlockAllocatorEdgeCases:
    """Test edge cases and error conditions."""

    @staticmethod
    def test_oom_behavior():
        """Test out-of-memory behavior."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=2,  # Small number for easy OOM
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # Allocate all blocks
        block1 = allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        block2 = allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        
        # Should be out of blocks now
        assert allocator.get_num_free_blocks(Device.GPU) == 0
        
        # Next allocation should raise exception
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)

    @staticmethod
    def test_zero_blocks():
        """Test behavior with zero blocks (edge case)."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=0,
            num_cpu_blocks=5,  # Ignored
            block_size=4,
        )
        
        assert allocator.get_num_total_blocks(Device.GPU) == 0
        assert allocator.get_num_free_blocks(Device.GPU) == 0
        
        # Should immediately fail to allocate
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)

    @staticmethod
    @pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
    def test_block_reuse_after_free(allocator_type: str):
        """Test that blocks can be reused after being freed."""
        allocator = UnifiedBlockAllocator.create(
            allocator_type=allocator_type,
            num_gpu_blocks=5,
            num_cpu_blocks=0,
            block_size=4,
        )
        
        # Allocate a block
        original_block = allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        original_id = original_block.block_id
        
        # Free the block
        allocator.free(original_block)
        assert original_block.block_id is None
        
        # Allocate a new block - should reuse the ID
        new_block = allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        assert new_block.block_id == original_id