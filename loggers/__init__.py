"""The metrics subfolder.

Implements the LogFn class that computes metrics during training.
"""

from loggers.base import Logger, LogState, LogMetrics
from loggers.base import get_internal_logs
from loggers.demo import simple_log
from loggers.demo import full_log