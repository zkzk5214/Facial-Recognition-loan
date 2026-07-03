import os
import time
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class MultiCompatibleTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A custom log handler that extends TimedRotatingFileHandler to ensure that log rotation works correctly 
    even if the application is restarted. It computes rollover times based on the current time and ensures 
    that log files are rotated at the expected intervals, handling edge cases like DST changes.
    """

    def computeRollover(self, currentTime: int) -> int:
        """Compute the rollover time based on the current time."""
        # Normalize rollover anchor to the formatted suffix boundary.
        t_str = time.strftime(self.suffix, time.localtime(currentTime))
        t = int(time.mktime(time.strptime(t_str, self.suffix)))
        return super().computeRollover(t)

    def doRollover(self):
        """Perform log rotation, ensuring the new log file is created after the old one is closed."""
        # Close the currently-open file stream before rotating.
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        # rolloverAt is the next rollover time, so compute the previous rollover time by subtracting the interval
        t = self.rolloverAt - self.interval 
        # If the handler is configured for UTC mode, use GMT for the rotated file's timestamp. 
        # No DST concerns in UTC.
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)

        # Build the destination filename.
        # Time rollovers keep the daily suffix; size rollovers use the current hour and minute.
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        if not os.path.exists(dfn):
            try:
                self.rotate(self.baseFilename, dfn)
            except FileNotFoundError:
                pass

        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

        # Reopen the stream for the new log file after rotation.        
        if not self.delay:
            self.stream = self._open()

        # Compute the next rollover time using the overridden computeRollover().
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval

        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow: 
                    addend = -3600
                else:
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt


def create_logger(
    log_path: str,
    logger_name: str,
    log_level: int = logging.INFO,
    backup_days: int = 30,
    rotate_at: str = "04:02",
) -> logging.Logger:
    """Create and return a configured ``logging.Logger`` with time-based rotation and optional size splitting."""
    # Convert to absolute path, then create all parent directories. 
    log_file_path = Path(log_path).resolve()
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    # Produces log lines like: 2026-06-10 04:02:15.123 [INFO] : VoxCPM engine ready
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    at_time = datetime.datetime.strptime(rotate_at, "%H:%M").time()
    handler = MultiCompatibleTimedRotatingFileHandler(
        str(log_file_path),
        when="midnight",
        atTime=at_time,
        backupCount=backup_days,
    )
    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    logger = logging.getLogger(logger_name)
    logger.handlers.clear() # Remove any previously attached handlers.
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.propagate = False # Prevents log messages from being passed to parent loggers.

    return logger
