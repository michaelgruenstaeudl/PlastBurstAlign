from __future__ import annotations

import os
import shutil
import tempfile
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple


def split_list(input_list: List, num_lists: int) -> List[List]:
    """
    From a source list, creates a list of lists of the same elements of the original list,
    where each inner list is the length of the specified number. If the length of the original
    list is not evenly divided by the specified number, the last inner list will have a length
    less than this number.

    This provides a similar function as `itertools.batched()`, which
    was added in Python 3.12.

    Args:
        input_list: List of objects.
        num_lists: Number of lists to create.

    Returns: List of lists.

    """
    # find the desired number elements in each list
    list_len = len(input_list) // num_lists

    # create the first num_lists-1 lists
    nested_list = [
        input_list[index * list_len : (index + 1) * list_len]
        for index in range(num_lists - 1)
    ]
    # create the final list, including the division remainder number of elements
    nested_list.append(input_list[(num_lists - 1) * list_len :])
    return nested_list


def bounded_submit_iterator(
    executor: Executor,
    fn: Callable[..., Any],
    items: Iterable[Any],
    max_in_flight: Optional[int] = None,
) -> Iterator[Tuple[Any, Future]]:
    """
    Submits tasks to an executor in a bounded manner, maintaining at most
    max_in_flight tasks queued/running concurrently.

    Args:
        executor: An Executor instance (ThreadPoolExecutor or ProcessPoolExecutor).
        fn: Task callable that accepts one item from items.
        items: An iterable or generator of items to process.
        max_in_flight: Maximum active/queued futures allowed. Defaults to 2 * max_workers.

    Yields:
        Tuple of (item, future) as each task completes.
    """
    if max_in_flight is None:
        workers = getattr(executor, "_max_workers", 2) or 2
        max_in_flight = max(1, workers * 2)
    else:
        max_in_flight = max(1, max_in_flight)

    item_iter = iter(items)
    in_flight = {}

    for item in item_iter:
        fut = executor.submit(fn, item)
        in_flight[fut] = item
        if len(in_flight) >= max_in_flight:
            break

    try:
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                item = in_flight.pop(fut)
                try:
                    next_item = next(item_iter)
                    new_fut = executor.submit(fn, next_item)
                    in_flight[new_fut] = next_item
                except StopIteration:
                    pass
                yield item, fut
    finally:
        for fut in in_flight:
            fut.cancel()


def ordered_bounded_map(
    executor: Executor,
    fn: Callable[..., Any],
    items: Iterable[Any],
    window: Optional[int] = None,
) -> Iterator[Tuple[Any, Future]]:
    """Runs tasks on an executor and yields them strictly in submission order.

    Unlike `bounded_submit_iterator`, which yields futures as they complete, the
    results here are handed out in the order the items were submitted, so that
    downstream consumers (for example database commits) are independent of task
    completion order. At most `window` items are submitted-but-not-yet-yielded at
    any time, so both the executor queue and the buffer of finished-but-waiting
    results stay bounded (head-of-line blocking can only stall the workers for
    the last `window - 1` items).

    Args:
        executor: An Executor instance (ThreadPoolExecutor or ProcessPoolExecutor).
        fn: Task callable that accepts one item from items.
        items: An iterable or generator of items to process.
        window: Maximum number of outstanding items. Defaults to 2 * max_workers.

    Yields:
        Tuple of (item, completed future) in submission order.
    """
    if window is None:
        workers = getattr(executor, "_max_workers", 2) or 2
        window = max(1, workers * 2)
    else:
        window = max(1, window)

    item_iter = iter(items)
    pending: deque[Tuple[Any, Future]] = deque()
    exhausted = False
    try:
        while True:
            while not exhausted and len(pending) < window:
                try:
                    item = next(item_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((item, executor.submit(fn, item)))
            if not pending:
                return
            item, fut = pending.popleft()
            wait([fut])
            yield item, fut
    finally:
        for _, fut in pending:
            fut.cancel()


def resolve_scratch_dir(preferred: str | None = None) -> str:
    """Resolves the highest-speed scratch directory available on the host.

    Order of precedence:
      1. Explicitly preferred path (if provided and writable)
      2. SLURM_TMPDIR (Node-local NVMe on SLURM clusters)
      3. LOCAL_SCRATCH / TMPDIR
      4. /dev/shm (POSIX RAM disk, only if >= 8 GB free space available)
      5. System tempfile fallback
    """
    candidates = [
        preferred,
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("LOCAL_SCRATCH"),
        os.environ.get("TMPDIR"),
    ]

    for path in candidates:
        if path and os.path.exists(path) and os.access(path, os.W_OK):
            return path

    # Check /dev/shm for sufficient capacity
    if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
        try:
            usage = shutil.disk_usage("/dev/shm")
            if usage.free >= 8 * (1024**3):  # at least 8 GB free
                return "/dev/shm"
        except OSError:
            pass

    return tempfile.gettempdir()
