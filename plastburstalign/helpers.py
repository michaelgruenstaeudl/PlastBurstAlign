from concurrent.futures import Executor, Future, wait, FIRST_COMPLETED
from typing import List, Callable, Iterable, Iterator, Tuple, Any, Optional


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
        input_list[index * list_len : (index + 1) * list_len] for index in range(num_lists - 1)
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
