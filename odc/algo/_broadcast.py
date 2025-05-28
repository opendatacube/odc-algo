# This file is part of the Open Data Cube, see https://opendatacube.org for more information
#
# Copyright (c) 2015-2025 ODC Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dask Distributed Tools.

- pool_broadcast
"""

from __future__ import annotations

from random import randint
from typing import Any

from dask.distributed import Client, Queue


def _bcast_action(
    q1: Queue, q2: Queue, tk: int, action: Any, args: list[Any], kwargs: dict[str, Any]
) -> Any:
    """

    :param q1: Will put to ``tk`` into this queue first
    :param q2: Will get an item from this queue after completing action
    :param tk: Token to identify this task
    :param action: Callable
    :param args: Ordered arguments to action
    :param kwargs: Named arguments to action

    :returns: result of calling action(*args, **kwargs)
    """
    q1.put(tk)  # tell main thread we started
    try:
        x = action(*args, **kwargs)
    finally:
        # wait for all threads to start
        # (q2 is expected to be empty until q1 is filled)
        q2.get()
    return x


def pool_broadcast(
    client: Client, action: Any, *args: list[Any], **kwargs: dict[str, Any]
):
    """Call ``action(*args, **kwargs)`` on every worker thread.

    This function block until all tasks are complete, expectation is
    that this is called at the very beginning on an empty pool, if called
    on a busy pool this will block until all active tasks are complete.

    Broadcast is achieved by blocking every task until all tasks have started,
    every worker does the following:

    1. Let the primary task know this task has started
    2. Perform action
    3. Wait for all other tasks to start
    4. Finish

    Steps (1) and (3) are achieved using distributed Queues, step (1) is a
    non-blocking ``put`` and step (3) is a blocking ``get``.

    :param client: Dask client object
    :param action: Callable `action(*args, **kwargs)`
    :param args: Ordered arguments to action
    :param kwargs: Named arguments to action

    """
    postfix = f"-{randint(0, 1 << 64):02x}"
    total_worker_threads = sum(client.ncores().values())
    q1 = Queue("q1" + postfix, client=client, maxsize=total_worker_threads)
    q2 = Queue("q2" + postfix, client=client, maxsize=total_worker_threads)

    ff = [
        client.submit(
            _bcast_action,
            q1,
            q2,
            i,
            action,
            args,
            kwargs,
            key=f"broadcast_action_{i:04d}{postfix}",
        )
        for i in range(total_worker_threads)
    ]

    tks = set()
    for _ in range(total_worker_threads):
        tks.add(q1.get())  # blocking

    assert len(tks) == total_worker_threads

    # at this point all workers have launched
    # allow them to continue
    for i in range(total_worker_threads):
        q2.put(i)  # should not block

    # block until all done and return result
    return [f.result() for f in ff]
