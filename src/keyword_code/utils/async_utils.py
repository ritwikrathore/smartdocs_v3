"""
Async utility functions for the keyword_code package.
"""

import asyncio
import concurrent.futures
from typing import List, Callable, Any
from ..config import logger


def run_async(coro):
    """Helper function to run async code in a thread-safe manner."""
    try:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_running():
            # If a loop is running, create a future and run in executor
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        logger.info("No current event loop found, creating a new one.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f"Error running async task: {e}", exc_info=True)
        raise


async def run_in_threadpool(func, *args, **kwargs):
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, lambda: func(*args, **kwargs)
        )


async def run_tasks_in_parallel(tasks: List[Callable[[], Any]], max_workers: int) -> List[Any]:
    """
    Run multiple tasks in parallel using a thread pool.

    Args:
        tasks: List of callable functions to execute
        max_workers: Maximum number of workers to use

    Returns:
        List of results from the tasks
    """
    if not tasks:
        return []

    # Use the minimum of tasks length and max_workers to avoid creating unnecessary workers
    actual_workers = min(len(tasks), max_workers)
    logger.info(f"Running {len(tasks)} tasks with {actual_workers} workers")

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all tasks to the executor
        futures = [loop.run_in_executor(executor, task) for task in tasks]

        # Wait for all tasks to complete
        results = await asyncio.gather(*futures, return_exceptions=True)

        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with error: {result}")

        return results
