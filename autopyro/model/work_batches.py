"""Let interactive threads run between batches of large model operations."""
from itertools import batched
from threading import current_thread, main_thread
from time import sleep


def cooperative_items(items):
    """Give the GUI a scheduling window while a worker traverses a large collection."""
    if len(items) < 8192 or current_thread() is main_thread():
        return items

    def iterate():
        for batch in batched(items, 2048):
            yield from batch
            # Release the GIL even when the Qt thread is waiting in native code.
            sleep(0.001)

    return iterate()
