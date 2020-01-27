# -*- coding: UTF-8 -*-
import logging
import os
from typing import Optional
from datetime import datetime
import sys

class Logger:
    def __init__(self, debug: bool = False):
        self.sym_error = '❌'
        self.sym_success = '✅'
        self.sym_result = '➡️'
        self.sym_tip = '💡'
        self.sym_warning = '⚠️'
        self.sym_important = '❗'
        self.sym_save = '💾'

        self._setup(debug)

    def _setup(self, debug: bool):
        # Create a log file
        current_datetime = datetime.now()
        full_datetime_stamp = current_datetime.strftime("%d_%m_%Y-%H_%M_%S")
        current_date_stamp = current_datetime.strftime("%d_%m_%Y")
        log_filename = 'results/logs/{}/run_{}.log'.format(current_date_stamp, full_datetime_stamp)

        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        file_handler = logging.FileHandler(filename=log_filename, encoding='utf-8')
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers = [stdout_handler]
        logging.basicConfig(handlers=handlers, format='\n%(asctime)s - %(levelname)s  - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    def info(self, message, next_step: Optional[str] = None):
        log_str = ""
        log_str += f"{message} \n"

        if next_step:
            log_str += f"\n\t {self.sym_result} {next_step}"

        logging.info(log_str)

    def error(self, message, next_step: Optional[str] = None, tip: Optional[str] = None):
        log_str = ""
        log_str += f" {str(self.sym_error)} {message} \n"

        if next_step:
            log_str += f"\n\t {self.sym_result} {next_step}"

        if tip:
            log_str += f"\n\t {self.sym_tip} {tip}"

        logging.error(log_str)

    def success(self, message, next_step: Optional[str] = None):
        log_str = ""
        log_str += f"{str(self.sym_success)} {message} \n"

        if next_step:
            log_str += f"\n\t {self.sym_result} {next_step}"

        logging.info(log_str)

    def warning(self, message, next_step: Optional[str] = None, tip: Optional[str] = None):
        log_str = ""
        log_str += f" {str(self.sym_warning)} {message} \n"

        if next_step:
            log_str += f"\n\t {self.sym_result} {next_step}"

        if tip:
            log_str += f"\n\t {self.sym_tip} {tip}"

        logging.error(log_str)

    def important(self, message):
        log_str = ""
        log_str += f" {str(self.sym_important)} {message} \n"

        logging.info(log_str)

    def save(self, message):
        log_str = ""
        log_str += f" {str(self.sym_save)} {message} \n"

        logging.info(log_str)

logger = Logger()
