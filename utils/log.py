import os
import re
import time
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List


class MultiCompatibleTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    A custom log handler that extends TimedRotatingFileHandler to ensure that log rotation works correctly 
    even if the application is restarted. It computes rollover times based on the current time and ensures 
    that log files are rotated at the expected intervals, handling edge cases like DST changes.
    """

    maxBytes: int
    _size_suffix = "%Y-%m-%d_%H-%M"
    _cleanup_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:_\d{2}-\d{2})?(?:\.\d+)?$")

    def __init__(self, filename: str, maxBytes: int = 0, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        self.maxBytes = maxBytes

    def computeRollover(self, currentTime: int) -> int:
        """Compute the rollover time based on the current time."""
        # Normalize rollover anchor to the formatted suffix boundary.
        t_str = time.strftime(self.suffix, time.localtime(currentTime))
        t = int(time.mktime(time.strptime(t_str, self.suffix)))
        return super().computeRollover(t)

    def shouldRollover(self, record: logging.LogRecord) -> int:
        """Trigger rollover when either the scheduled time or the file size limit is reached."""
        if int(time.time()) >= self.rolloverAt:
            return 1

        max_bytes = getattr(self, "maxBytes", 0)
        if max_bytes <= 0:
            return 0

        if self.stream is None:
            self.stream = self._open()
            self.stream.seek(0, os.SEEK_END)

        message = "%s\n" % self.format(record)
        try:
            current_size = os.stat(self.baseFilename).st_size
        except FileNotFoundError:
            current_size = 0

        encoding = self.encoding or "utf-8"
        message_size = len(message.encode(encoding, errors="replace"))
        if current_size + message_size >= max_bytes:
            return 1
        return 0

    def _build_rollover_filename(self, suffix: str) -> str:
        """Ensure rollover filenames stay unique when multiple rotations happen with the same timestamp."""
        base_name = self.baseFilename + "." + suffix
        if not os.path.exists(base_name):
            return base_name

        index = 1
        while True:
            candidate = f"{base_name}.{index}"
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def getFilesToDelete(self) -> List[str]:
        """Delete old rotated files, including size-split files that share the same day prefix."""
        if self.backupCount <= 0:
            return []

        dir_name, base_name = os.path.split(self.baseFilename)
        prefix = base_name + "."
        files_by_day: Dict[str, List[str]] = {}

        for file_name in os.listdir(dir_name):
            if not file_name.startswith(prefix):
                continue

            suffix = file_name[len(prefix):]
            match = self._cleanup_pattern.match(suffix)
            if not match:
                continue

            day_key = match.group(1)
            files_by_day.setdefault(day_key, []).append(os.path.join(dir_name, file_name))

        if len(files_by_day) <= self.backupCount:
            return []

        stale_days = sorted(files_by_day)[:-self.backupCount]
        files_to_delete: List[str] = []
        for day_key in stale_days:
            files_to_delete.extend(sorted(files_by_day[day_key]))
        return files_to_delete

    def doRollover(self):
        """Perform log rotation, ensuring the new log file is created after the old one is closed."""
        # Close the currently-open file stream before rotating.
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        is_time_rollover = currentTime >= self.rolloverAt
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
        if is_time_rollover:
            suffix = time.strftime(self.suffix, timeTuple)
        else:
            size_time_tuple = time.gmtime(currentTime) if self.utc else time.localtime(currentTime)
            suffix = time.strftime(self._size_suffix, size_time_tuple)

        dfn = self._build_rollover_filename(suffix)
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
    max_bytes: int = 0,
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
        maxBytes=max_bytes,
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
