import logging
import logging.config


def test_logger_empty_init(mocker):
    log_config_spy = mocker.spy(logging.config, "dictConfig")
    from guardrails.logger import logger

    assert logger.level == logging.NOTSET
    # It's not possible to retrieve the logging configuration
    # in an easily machine readable format
    # Instead spy on logging.config.dictConfig
    assert log_config_spy.call_count == 0


def test_logger_is_singleton():
    from guardrails.logger import get_logger, logger

    logger_1 = get_logger()
    logger_2 = get_logger()
    assert logger == logger_1
    assert logger_1 == logger_2


def test_set_level(mocker):
    from guardrails.logger import logger, logger_config, set_level

    set_level(logging.NOTSET)

    setLevel_spy = mocker.spy(logger, "setLevel")

    set_level(logging.WARNING)

    assert logger.level == logging.WARNING
    assert logger_config.level == logging.WARNING
    assert setLevel_spy.call_count == 1

    set_level(logging.NOTSET)


def test_set_config(mocker):
    log_config_spy = mocker.patch("logging.config.dictConfig")
    from guardrails.logger import set_config

    new_config = {"version": 1}
    set_config(new_config)

    assert log_config_spy.call_count == 1


def test_set_scope():
    from guardrails.logger import logger, set_level

    assert logger.level == logging.NOTSET
    set_level(logging.INFO)
    assert logger.level == logging.INFO


"""
Since tests run in parallel, we need a distinct logger and handler instance
for this test in order for the assertions to work.
"""


def test_scope_handler():
    from guardrails.logger import ScopeHandler, base_scope

    test_logger = logging.getLogger("test_scope_handler")
    test_logger.setLevel(logging.INFO)
    new_handler = ScopeHandler()
    test_logger.addHandler(new_handler)

    test_logger.info("test log 1")

    new_handler.set_scope("test-2")

    test_logger.warn("test log 2")

    base_logs = new_handler.get_logs(base_scope)
    assert len(base_logs) == 1
    assert base_logs[0].getMessage() == "test log 1"

    test_2_logs = new_handler.get_logs("test-2")
    assert len(test_2_logs) == 1
    assert test_2_logs[0].getMessage() == "test log 2"

    all_logs = new_handler.get_all_logs()
    assert len(all_logs) == 2
    assert all_logs[0].getMessage() == "test log 1"
    assert all_logs[1].getMessage() == "test log 2"
