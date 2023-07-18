from multiprocessing import Process, Queue
from time import perf_counter, sleep

from mjregrasping.homotopy_regrasp_planner import HomotopyRegraspPlanner
from mjregrasping.physics import Physics


def peek(q: Queue):
    result = q.get()
    q.put(result)
    return result


class AsyncPlanner:

    def __init__(self, planner: HomotopyRegraspPlanner):
        self.planner = planner
        self.viz = viz
        self.done = False
        self.thread = Process(target=self.generate_in_thread)
        self.phy_queue = Queue()
        self.results_queue = Queue()
        self.locs = None
        self.subgoals = None

        self.thread.start()

    def generate_in_thread(self):
        print("Starting regrasp planning thread...")

        # Wait for the first phy object to be received
        phy = None  # just to suppress linting errors
        while self.phy_queue.empty():
            sleep(0.1)

        while not self.done:
            # Get the latest phy object and empty the queue
            while not self.phy_queue.empty():
                phy = self.phy_queue.get()

            t0 = perf_counter()
            # NOTE: viz across multiple processes doesn't work :(
            locs, subgoals, _, _ = self.planner.generate(phy, viz=None)
            dt = perf_counter() - t0

            print(f'new locs: {self.locs}')
            print(f'dt: {dt:.3f}')

            self.results_queue.put((locs, subgoals))

            sleep(1.0)  # this must be greater than the sleep in `get_latest`

    def update_phy(self, phy: Physics):
        # copy because we don't want to other modifications to phy to effect the phy used in this class
        self.phy_queue.put(phy.copy_all())

    def get_latest(self):
        """
        Returns: The most recently computed locs and subgoals. If none have been computed yet, waits until they are.
        """
        while self.results_queue.empty():
            sleep(0.1)

        locs, subgoals = peek(self.results_queue)

        return locs, subgoals

    def stop(self):
        self.done = True
        self.thread.join()
