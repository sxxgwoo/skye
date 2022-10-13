# python
import time
import logging


logger = logging.getLogger(__name__)

def time_usage(func):
    """함수의 시간 비용을 측정하는 데코레이터

    사용 예:
    ```py
    @time_usage
    def some_loop():
        for i in range(1000):
            i ** i
    ```

    Args:
        func ([type]): [description]
    """
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        logger.info("{0} {1} {2}".format(func.__module__, func.__name__, "elapsed time: %fs" % (end_ts - beg_ts)))
        return retval
    return wrapper