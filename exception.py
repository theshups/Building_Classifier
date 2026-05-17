"""exception.py - Custom exception with traceback detail."""
import sys
from logger import get_logger
log = get_logger(__name__)

def _detail(error, tb_sys) -> str:
    _, _, tb = tb_sys.exc_info()
    if tb is None:
        return str(error)
    return f"[{tb.tb_frame.f_code.co_filename}] line {tb.tb_lineno}: {error}"

class AppException(Exception):
    def __init__(self, error, error_sys):
        super().__init__(str(error))
        self.error_message = _detail(error, error_sys)
        log.error(self.error_message)
    def __str__(self):
        return self.error_message
