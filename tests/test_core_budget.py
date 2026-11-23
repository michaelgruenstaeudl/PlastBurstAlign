from __future__ import annotations

import unittest

from plastburstalign.task_queue import CoreBudgetPool


class TestCoreBudgetPool(unittest.TestCase):
    def test_initial_state(self):
        pool = CoreBudgetPool(total_cores=16, total_tasks=50)
        self.assertEqual(pool.total_cores, 16)
        self.assertEqual(pool.available_cores, 16)
        self.assertEqual(pool.remaining, 50)

    def test_deep_queue_leases_single_core(self):
        pool = CoreBudgetPool(total_cores=16, total_tasks=100)
        lease = pool.lease_cores(file_size=1000)
        self.assertEqual(lease, 1)
        self.assertEqual(pool.available_cores, 15)
        pool.release_cores(lease)
        self.assertEqual(pool.available_cores, 16)

    def test_tail_straggler_fair_share_expansion(self):
        # Only 2 tasks remaining with 16 cores available
        pool = CoreBudgetPool(total_cores=16, total_tasks=2)
        lease1 = pool.lease_cores(file_size=1000)
        self.assertEqual(lease1, 8)  # 16 // 2 = 8
        self.assertEqual(pool.available_cores, 8)

        pool.task_completed()  # remaining drops to 1
        lease2 = pool.lease_cores(file_size=1000)
        self.assertEqual(lease2, 8)  # Remaining available cores
        self.assertEqual(pool.available_cores, 0)

        pool.release_cores(lease1)
        pool.release_cores(lease2)
        self.assertEqual(pool.available_cores, 16)

    def test_massive_file_preferred_heuristic(self):
        pool = CoreBudgetPool(total_cores=16, total_tasks=50)
        # On a 16-core pool, a 50 MB file requests 16 preferred cores
        lease = pool.lease_cores(file_size=50 * 1024 * 1024)
        self.assertEqual(lease, 16)
        self.assertEqual(pool.available_cores, 0)
        pool.release_cores(lease)

    def test_context_manager_exception_safety(self):
        pool = CoreBudgetPool(total_cores=8, total_tasks=1)
        try:
            with pool.lease(file_size=0) as leased:
                self.assertEqual(leased, 8)
                self.assertEqual(pool.available_cores, 0)
                raise RuntimeError("Simulated worker error")
        except RuntimeError:
            pass

        # Cores must be fully restored upon exiting the context block
        self.assertEqual(pool.available_cores, 8)


if __name__ == "__main__":
    unittest.main()
