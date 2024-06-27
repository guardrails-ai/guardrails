import asyncio
import concurrent.futures
import random
import sys
import time
from multiprocessing import Pool

from guardrails.guard_call_logging import SyncTraceHandler


DELAY = 0.1
hoisted_logger = SyncTraceHandler()


def main(num_threads: int, num_log_messages: int):
    log_levels = list()
    for _ in range(num_log_messages):
        log_levels.append(random.randint(0, 5))
    print("Multiprocessing: Trying with hoisted logger:", end=" ")
    with Pool(num_threads) as pool:
        pool.map(log_with_hoisted_logger, log_levels)
    print("Done.")
    print("Multiprocessing: Trying with acquired logger:", end=" ")
    with Pool(num_threads) as pool:
        pool.map(log_with_acquired_singleton, log_levels)
    print("Done.")
    print("Multithreading: Trying with hoisted logger:", end=" ")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for level in log_levels:
            out = executor.submit(log_with_hoisted_logger, level)
            out.result()
        #executor.map(log_with_hoisted_logger, log_levels)
    print("Done")
    print("Multithreading: Trying with acquired logger:", end=" ")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for level in log_levels:
            out = executor.submit(log_with_acquired_singleton, level)
            out.result()
        #executor.map(log_with_acquired_singleton, log_levels)
    print("Done")
    print("Asyncio: Trying with hoisted logger:", end=" ")
    async def do_it(level: int):
        log_with_hoisted_logger(level)
    async def do_it_again(level: int):
        log_with_acquired_singleton(level)
    for level in log_levels:
        asyncio.run(do_it(level))
    for level in log_levels:
        asyncio.run(do_it_again(level))
    print("Done")


def log_with_hoisted_logger(log_level: int):
    start = time.time()
    end = time.time()
    hoisted_logger.log(
        "hoisted_logger",
        start,
        end,
        "Kept logger from hoisted.",
        "Success.",
        "",
        log_level
    )
    time.sleep(DELAY)


def log_with_acquired_singleton(log_level: int):
    # Try grabbing a reference to the sync writer.
    start = time.time()
    log = SyncTraceHandler()
    end = time.time()
    log.log(
        "acquired_logger",
        start,
        end,
        "Got logger with acquired singleton.",
        "It worked.",
        "",
        log_level
    )
    time.sleep(DELAY)


if __name__ == '__main__':
    if "--help" in sys.argv:
        print("Optional args: --num_threads, --num_log_messages")
    else:
        thread_count = 4
        try:
            num_threads_arg_pos = sys.argv.index('--num_threads')
            if num_threads_arg_pos != -1:
                thread_count = int(sys.argv[num_threads_arg_pos + 1])
        except Exception:
            pass
        log_message_count = 10
        try:
            num_log_messages_pos = sys.argv.index('--num_log_messages')
            if num_log_messages_pos != -1:
                log_message_count = int(sys.argv[num_log_messages_pos + 1])
        except Exception:
            pass

        main(num_threads=thread_count, num_log_messages=log_message_count)
