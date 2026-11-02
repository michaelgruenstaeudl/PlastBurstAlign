import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from plastburstalign.helpers import bounded_submit_iterator


class BoundedTaskQueueTests(unittest.TestCase):
    def test_bounded_in_flight_concurrency_limit(self):
        max_in_flight = 3
        active_lock = threading.Lock()
        current_active = 0
        peak_active = 0

        def simulate_work(item_id: int):
            nonlocal current_active, peak_active
            with active_lock:
                current_active += 1
                if current_active > peak_active:
                    peak_active = current_active
            time.sleep(0.02)
            with active_lock:
                current_active -= 1
            return item_id * 2

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = []
            for item, future in bounded_submit_iterator(
                executor, simulate_work, range(15), max_in_flight=max_in_flight
            ):
                results.append((item, future.result()))

        self.assertEqual(len(results), 15)
        self.assertLessEqual(peak_active, max_in_flight)
        self.assertEqual(sorted(dict(results).keys()), list(range(15)))

    def test_empty_and_single_item_iterables(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            empty_results = list(bounded_submit_iterator(executor, lambda x: x, []))
            self.assertEqual(empty_results, [])

            single_results = [
                fut.result() for _, fut in bounded_submit_iterator(executor, lambda x: x * 10, [42])
            ]
            self.assertEqual(single_results, [420])

    def test_exception_propagation(self):
        def faulty_work(n: int):
            if n == 2:
                raise ValueError("Planned error")
            return n

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = {}
            for item, future in bounded_submit_iterator(
                executor, faulty_work, range(5), max_in_flight=2
            ):
                try:
                    results[item] = future.result()
                except ValueError as e:
                    results[item] = str(e)

        self.assertEqual(results[2], "Planned error")
        self.assertEqual(results[0], 0)
        self.assertEqual(results[1], 1)
        self.assertEqual(results[3], 3)
        self.assertEqual(results[4], 4)

    def test_early_break_cancels_remaining_tasks(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            collected = []
            for item, future in bounded_submit_iterator(
                executor, lambda x: x, range(20), max_in_flight=3
            ):
                collected.append(future.result())
                if len(collected) == 3:
                    break
            self.assertEqual(len(collected), 3)


if __name__ == "__main__":
    unittest.main()
