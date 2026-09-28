# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import errno
import socket
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.distributed.utils import StatelessProcessGroup


def exchange_metadata(host, port, rank, listener):
    group = StatelessProcessGroup.create(
        host, port, rank, 2, store_timeout=10, listen_socket=listener
    )
    group.store.set(f"rank-{rank}", f"value-{rank}".encode())
    assert [group.store.get(f"rank-{peer}") for peer in range(2)] == [
        b"value-0",
        b"value-1",
    ]
    data = {"step": 1, "names": ["weight"]}
    assert group.broadcast_obj(data if rank == 0 else None, src=0) == data
    group.barrier(timeout=10)


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
@pytest.mark.parametrize("prebound", [False, True])
def test_stateless_process_group_tcp_listener(host, prebound):
    """Exchange metadata with internal and caller-owned TCP listeners."""
    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    try:
        listener = socket.socket(family, socket.SOCK_STREAM)
    except OSError as error:
        if host == "::1" and error.errno == errno.EAFNOSUPPORT:
            pytest.skip("IPv6 sockets are unavailable")
        raise
    with listener:
        try:
            listener.bind((host, 0))
        except OSError as error:
            if host == "::1" and error.errno == errno.EADDRNOTAVAIL:
                pytest.skip("IPv6 loopback is unavailable")
            raise
        port = listener.getsockname()[1]
        if prebound:
            listener.listen()
        else:
            listener.close()
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = [
                executor.submit(
                    exchange_metadata,
                    host,
                    port,
                    rank,
                    listener if prebound and rank == 0 else None,
                )
                for rank in range(2)
            ]
            for future in pending:
                future.result(timeout=30)
        if prebound:
            assert listener.fileno() == -1
