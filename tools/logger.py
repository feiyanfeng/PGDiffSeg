import os
import logging

class Logger:
    def __init__(self, filename='log.txt', show=True):
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)

        self.logger = logging.getLogger('custom_logger')
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(filename)
        self.handler.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        self.show = show

    def info(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.info(message)
        if format:
            self.handler.setFormatter(self.formatter)
        if self.show:
            print(message)


    def error(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.error(message)
        if format:
            self.handler.setFormatter(self.formatter)
        if self.show:
            print(message)

    
    def warning(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.warning(message)
        if format:
            self.handler.setFormatter(self.formatter)
        if self.show:
            print(message)


    def critical(self, message, format=None):
        if format:
            self.handler.setFormatter(logging.Formatter(format))
        self.logger.critical(message)
        if format:
            self.handler.setFormatter(self.formatter)
        if self.show:
            print(message)

            
    def close(self):
        logging.shutdown()

    def __del__(self):
        self.close()

# # 使用示例
# logger = Logger('log/my_log.log')

# logger.info('This is an info log message.')
# logger.error('This is an error log message.', '%(levelname)s - %(message)s')
# logger.warning('This is a warning log message.')
# logger.critical('This is a critical log message.\n\n\n')

