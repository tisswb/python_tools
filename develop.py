import time


def fn_timer(logger=None):
    """记录函数或方法的运行时间的修饰器

    Args:
        logger (Logger, optional): 日志记录器实例. Defaults to None.
    """
    def timed(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            output = f"Execution Time: {execution_time:.3f} s, Function: {func.__name__}."
            if logger:
                logger.info(output)
            else:
                print(output)
            return result

        return wrapper

    return timed
