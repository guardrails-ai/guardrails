import asyncio
import concurrent.futures
import time
from multiprocessing import Pool, Process

from guardrails.call_tracing import TraceHandler

NUM_THREADS = 4

STOCK_MESSAGES = [
    "Lorem ipsum dolor sit amet",
    "consectetur adipiscing elit",
    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam",
    "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat",
    "Excepteur sint occaecat cupidatat non proident,",
    "sunt in culpa qui officia deserunt mollit anim id est laborum.",
]


# This is hoisted for testing to see how well we share.
_trace_logger = TraceHandler()


def test_multiprocessing_hoisted():
    """Preallocate a shared trace handler and try to log from multiple
    subprocesses."""
    with Pool(NUM_THREADS) as pool:
        pool.map(_hoisted_logger, ["multiproc_hoist" + msg for msg in STOCK_MESSAGES])


def test_multiprocessing_acquired():
    with Pool(NUM_THREADS) as pool:
        pool.map(_acquired_logger, ["multiproc_acq" + msg for msg in STOCK_MESSAGES])


def test_multithreading_hoisted():
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for msg in STOCK_MESSAGES:
            out = executor.submit(_hoisted_logger, "multithread_hoist" + msg)
            out.result()
        # executor.map(log_with_hoisted_logger, log_levels)


def test_multithreading_acquired():
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for msg in STOCK_MESSAGES:
            out = executor.submit(_acquired_logger, "multithread_acq" + msg)
            out.result()


def test_clear_logs_while_logging():
    # Hammer writes to the logfile while clearing from a separate process.
    # See if anything crashes.
    p = Process(target=_write_to_logs_a_bunch, args=(256, 0.01))
    p.start()
    trace_logger = TraceHandler()
    for _ in range(256):
        trace_logger.clear_logs()
        time.sleep(0.01)
    p.join()


def test_asyncio_hoisted():
    async def do_it(msg: str):
        _hoisted_logger(msg)

    for m in STOCK_MESSAGES:
        asyncio.run(do_it("async_hoisted" + m))


def test_asyncio_acquired():
    async def do_it_again(msg: str):
        _acquired_logger(msg)

    for m in STOCK_MESSAGES:
        asyncio.run(do_it_again("async_acq" + m))


def _hoisted_logger(msg: str):
    _trace_logger.log(
        "hoisted",
        time.time(),
        time.time(),
        "Testing the behavior of a hoisted logger.",
        msg,
        "",
    )


def _acquired_logger(msg):
    # Note that the trace logger is acquired INSIDE the method:
    start = time.time()
    trace_logger = TraceHandler()
    end = time.time()
    trace_logger.log("acquired", start, end, "Testing behavior of an acquired logger.", msg, "")


def _write_to_logs_a_bunch(count: int, delay: float):
    trace_logger = TraceHandler()
    for i in range(count):
        trace_logger.log(
            "acquired",
            time.time(),
            time.time(),
            f"Writing message {i} of {count} with {delay} seconds between them.",
            "",
            "",
        )
        time.sleep(delay)
