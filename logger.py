'''
    This script will log usefull information

    Usage : no direct call, used by train.py, train_with_hdf5.py and dataset.py.
    -------

    Main features :
    ---------------
        * Handle log level to get clear and readable log
        * Have helpers to deal with date and execution time

    Environment installation : Please refer to the README.md file.
    --------------------------

    Licence / feedback :
    --------------------
        Please feel free to reuse, ask me question or give me tips/comments. 
        I'm not an expert and would love to have some feed back and advise.

    @author Pierre Labadille
    @date 10/26/2017
    @version 1.0
    @todo Update to InceptionV4
'''

import sys
import time, datetime
import itertools

LOG_FILE = 'logs/info_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"

#- Helpers

def write_to_file(txt):
    with open(LOG_FILE, "a") as log:
        log.write(str(txt))

def log_level_in_tab(lvl):
    s = ""
    if lvl == 0:
        return s
    else:
        for _ in itertools.repeat(None, lvl):
            s += "\t"
    return s

def seconds_to_time_string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def datetime_difference_in_time_string(x, y):
    elapsedTime = y-x
    seconds = elapsedTime.total_seconds()
    return seconds_to_time_string(seconds)

#- Public functions

def header():
    s = "---------------------------------------------------------------------\n"
    s += "----New training started the " + str(datetime.datetime.now()) + "----\n"
    s += "---------------------------------------------------------------------\n"
    write_to_file(s)

def log(txt, lvl=0):
    write_to_file(log_level_in_tab(lvl) + txt + "\n")

def execution_time(start_time, title, lvl=0):
    end_time = time.time()
    seconds = end_time - start_time
    s = title + " : executed in " + seconds_to_time_string(seconds)
    log(s, lvl)


def footer(start_datetime):
    total_time = datetime_difference_in_time_string(start_datetime, datetime.datetime.now())
    s = "Script executed in " + total_time
    s += "\n---------------------------------------------------------------------\n"
    s += "---------------------------------------------------------------------\n\n"
    write_to_file(s)
