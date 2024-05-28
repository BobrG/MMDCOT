import io
from argparse import Namespace
import logging


class Logger(logging.Logger):
    @classmethod
    def prepare_logger(cls, loglevel='warning', logger_id='default_logger', logfile=None):
        loglevel = Logger.get_log_level(loglevel)
        logging.setLoggerClass(cls)
        logger = logging.getLogger(logger_id)
        logger.setLevel(loglevel)

        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(loglevel)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.info_stream = StreamHandler(logger, logging.INFO)

        if logfile is not None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(loglevel)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def log_args(self, args, args_name='Args'):
        if isinstance(args, Namespace):
            args = args.__dict__
        s = '\n'.join(f'--{k}: {v}' for k, v in args.items())
        self.info(f'{args_name}:\n{s}')

    @staticmethod
    def get_log_level(loglevel):
        if isinstance(loglevel, str):
            loglevel = {
                'critical': 50,
                'error': 40,
                'warning': 30,
                'info': 20,
                'debug': 10
            }[loglevel]

        if loglevel >= 50:
            return logging.CRITICAL
        elif loglevel >= 40:
            return logging.ERROR
        elif loglevel >= 30:
            return logging.WARNING
        elif loglevel >= 20:
            return logging.INFO
        elif loglevel >= 10:
            return logging.DEBUG
        elif loglevel >= 0:
            return [logging.WARNING, logging.INFO, logging.DEBUG][min(loglevel, 2)]
        else:
            return logging.NOTSET


class StreamHandler(io.StringIO):
    def __init__(self, logger, level):
        super().__init__()
        self.logger = logger
        self.level = level
        self.buf = ''

    def write(self, buf):
        self.buf = buf.strip('\r\n')

    def flush(self):
        self.logger.log(self.level, self.buf)
