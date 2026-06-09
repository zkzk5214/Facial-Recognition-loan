import os
import time
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler


class MultiCompatibleTimedRotatingFileHandler(TimedRotatingFileHandler):
    def computeRollover(self, currentTime: int) -> int:
        # Normalize rollover anchor to the formatted suffix boundary.
        t_str = time.strftime(self.suffix, time.localtime(currentTime))
        t = time.mktime(time.strptime(t_str, self.suffix))
        return super().computeRollover(t)

    def doRollover(self):
        # Close current stream first to avoid writing while rotating.
        if self.stream:
            self.stream.close()
            self.stream = None
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
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
            # Destination filename for the rotated log.
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)

        if not os.path.exists(dfn):
            try:
                self.rotate(self.baseFilename, dfn)
            except FileNotFoundError:
                pass

        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        # Compute next rollover and ensure it is strictly in the future.
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


class LoggerGenerator():
    def __init__(
        self,
        log_path,
        logger_name,
        log_level=logging.INFO,
        backup_days: int = 7,
        rotate_at: str = "04:02",
    ):
        # Ensure the target log directory exists.
        abs_log_path = os.path.abspath(log_path)
        log_dir_path = os.path.dirname(abs_log_path)

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d [%(levelname)s] : %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Rotate daily at a configurable HH:MM local time.
        at_time = datetime.datetime.strptime(rotate_at, "%H:%M")
        handler = MultiCompatibleTimedRotatingFileHandler(
            abs_log_path,
            when="midnight",
            atTime=at_time,
            backupCount=backup_days,
        )
        handler.setFormatter(formatter)
        handler.setLevel(log_level)

        logger = logging.getLogger(logger_name)
        # Rebuild handlers to avoid duplicate outputs on reinitialization.
        logger.handlers.clear()
        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.propagate = False

        self.logger = logger

    def get_logger(self):
        return self.logger

if __name__ == "__main__":
    log_path = "logs/test.log"
    logger = LoggerGenerator(log_path, "test_logger").get_logger()
    logger.info("This is a test log message.")
    