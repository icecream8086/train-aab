import os
# #
# #   useage
#     log_appender = LogAppender('/path/to/directory')
#     log_appender.append_log('This is a new log entry')
# #
# #

class LogAppender:
    def __init__(self, dir_path):
        self.dir_path = dir_path
    
    def append_log(self, text):
        log_file_path = os.path.join(self.dir_path, 'log.txt')
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                pass
        with open(log_file_path, 'a') as f:
            f.write(text + '\n')