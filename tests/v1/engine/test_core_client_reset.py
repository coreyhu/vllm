# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from unittest.mock import AsyncMock, call

import pytest

from vllm.v1.engine.core_client import DPLBAsyncMPClient


@pytest.fixture
def client():
    # Avoid starting engines: these checks exercise the utility RPC contract.
    instance = object.__new__(DPLBAsyncMPClient)
    instance.core_engines = [b"\x00\x00", b"\x01\x00", b"\x02\x00"]
    instance._kv_event_sources = {}
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_core", [None, 0, 1, 2])
@pytest.mark.parametrize("flags", [(False, False), (True, False), (True, True)])
async def test_reset_requires_success_from_every_core(client, failed_core, flags):
    client._call_utility_async = AsyncMock(
        side_effect=[index != failed_core for index in range(3)]
    )

    assert await client.reset_prefix_cache_async(*flags) is (failed_core is None)
    assert client._call_utility_async.await_args_list == [
        call("reset_prefix_cache", *flags, engine=engine)
        for engine in client.core_engines
    ]


@pytest.mark.asyncio
async def test_reset_waits_for_other_cores_before_raising(client):
    failed = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def reset(method, *args, engine):
        if engine == client.core_engines[0]:
            failed.set()
            raise RuntimeError("reset failed")
        await release.wait()
        completed.append(engine)
        return True

    client._call_utility_async = reset
    pending = asyncio.create_task(client.reset_prefix_cache_async())
    try:
        await asyncio.wait_for(failed.wait(), timeout=5)
        done, _ = await asyncio.wait([pending], timeout=0.05)
        assert not done
        release.set()
        with pytest.raises(RuntimeError, match="reset failed"):
            await asyncio.wait_for(pending, timeout=5)
        assert completed == client.core_engines[1:]
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_other_utilities_still_return_first_core_result(client):
    client._call_utility_async = AsyncMock(side_effect=["first", "second", "third"])
    assert await client.call_utility_async("other_utility") == "first"
    assert client._call_utility_async.await_count == len(client.core_engines)
