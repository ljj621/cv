import logging, time, os, torch


class Logger(object):
    def __init__(self, name, log_dir, log_file=None, log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
        
        if log_file is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = os.path.join(log_dir, f'{timestamp}.log')
        else:
            log_file = os.path.join(log_dir, log_file)
        
        file_handler = logging.FileHandler(log_file, 'w')
        stream_handler = logging.StreamHandler()
        handlers = [stream_handler, file_handler]
        for handler in handlers:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            self.logger.addHandler(handler)

        self.logger.propagate = False

        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)

        # self.reset()
    
    def info(self, msg):
        self.logger.info(msg)

    @staticmethod
    def get(name):
        logger = logging.getLogger(name)
        return logger

    



                
    






