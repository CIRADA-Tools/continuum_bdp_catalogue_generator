import os
import re
import time
import math
import numpy as np

def time_string_to_seconds(t_str):
    t_sec = 0
    try:
        h = int(t_str.split('h')[0] if re.search("h",t_str) else '0')
        t_str = re.sub(r"^\d+h","",t_str)
        m = int(t_str.split('m')[0] if re.search("\dm\d",t_str) else '0')
        t_str = re.sub(r"^\d+m","",t_str)
        s = int(t_str.split('s')[0] if re.search("\ds\d",t_str) else '0')
        t_str = re.sub(r"^\d+s","",t_str)
        ms = int(t_str.split('ms')[0] if re.search("ms",t_str) else '0')
        t_str = re.sub(r"^\d+ms","",t_str)
        us = int(t_str.split('us')[0] if re.search("us",t_str) else '0')
        t_sec = 3600*h+60*m+s+(ms+us/1000)/1000
    except Exception as e:
        print(f"ERR: {e}: {t_str}")
    return t_sec

def get_time_string(seconds):
    def f(n):
        return math.floor(n)
    t_str = ""
    if seconds > 0:
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        ms = int(1000.0*(seconds % 1))
        us = round(1000.0*((1000.0*seconds) % 1))
        t_str += ("%dh" % h) if f(h) > 0 else ""
        t_str += ("%dm" % m) if f(m) > 0 else ""
        t_str += ("%ds" % s) if f(s) > 0 else ""
        t_str += ("%dms" % ms) if f(ms) > 0 else ""
        t_str += ("%dus" % us) if f(us) > 0 else ""
    else:
        t_str += "0s"
    return t_str

class Timer:
    def __init__(self):
        self.start_time = None
        self.stop_time = None
        self.is_running = False

    def __get_time_string(self,seconds):
        #def f(n):
        #    return math.floor(n)
        #t_str = ""
        #if seconds > 0:
        #    m, s = divmod(seconds, 60)
        #    h, m = divmod(m, 60)
        #    ms = int(1000.0*(seconds % 1))
        #    us = round(1000.0*((1000.0*seconds) % 1))
        #    t_str += ("%dh" % h) if f(h) > 0 else ""
        #    t_str += ("%dm" % m) if f(m) > 0 else ""
        #    t_str += ("%ds" % s) if f(s) > 0 else ""
        #    t_str += ("%dms" % ms) if f(ms) > 0 else ""
        #    t_str += ("%dus" % us) if f(us) > 0 else ""
        #else:
        #    t_str += "0s"
        #return t_str
        return get_time_string(seconds)

    def start(self):
        self.is_running = True
        self.start_time = time.process_time()
        return self

    def stop(self):
        if self.is_running:
            self.stop_time = time.process_time()
            self.is_running = False
        return self

    def get_time(self,reset=True):
        if reset:
            self.stop()
        return self.stop_time - self.start_time

    def get_time_string(self,reset=True):
        return self.__get_time_string(self.get_time(reset))

    def print_processing_time(self):
        print(f"> Processing Time: {self.get_time_string()}")
        return self


class TimeStats:
    def __init__(self,max_counts=0,is_walk_clock_time=True):
        self.is_walk_clock_time = is_walk_clock_time
        self.max_counts = max_counts
        self.start_time = None
        self.time_i = None
        self.time_f = None
        self.deltas = list()
        self.stop_time = None
        self.is_running = False

    def __get_time_string(self,seconds):
        #def f(n):
        #    return math.floor(n)
        #t_str = ""
        #if seconds > 0:
        #    m, s = divmod(seconds, 60)
        #    h, m = divmod(m, 60)
        #    ms = int(1000.0*(seconds % 1))
        #    us = round(1000.0*((1000.0*seconds) % 1))
        #    t_str += ("%dh" % h) if f(h) > 0 else ""
        #    t_str += ("%dm" % m) if f(m) > 0 else ""
        #    t_str += ("%ds" % s) if f(s) > 0 else ""
        #    t_str += ("%dms" % ms) if f(ms) > 0 else ""
        #    t_str += ("%dus" % us) if f(us) > 0 else ""
        #else:
        #    t_str += "0s"
        #return t_str
        return get_time_string(seconds)

    def get_current_time(self):
        if self.is_walk_clock_time:
            return time.perf_counter()
        return time.process_time()

    def start(self):
        if not self.is_running:
            self.start_time = self.get_current_time()
            self.time_i = self.start_time
            self.time_f = self.start_time
            self.is_running = True
        return self

    def stop(self):
        if self.is_running:
            self.stop_time = self.get_current_time()
            self.is_running = False
        return self

    def update_dt(self):
        if not self.is_running:
            self.start()
        self.time_f = self.get_current_time()
        self.deltas.append(self.time_f-self.time_i)
        self.time_i = self.time_f
        return self

    def get_ellapsed_time(self,as_string=False):
        if self.is_running:
            dT = self.get_current_time()-self.start_time
        else:
            try:
                dT = time.stop_time-self.start_time
            except:
                dT = 0
        return self.__get_time_string(dT) if as_string else dT

    def get_avg(self,as_string=False):
        avg = np.average(self.deltas) if len(self.deltas) > 0 else 0
        return self.__get_time_string(avg) if as_string else avg

    def get_std(self,as_string=False):
        std = np.std(self.deltas) if len(self.deltas) > 0 else 0
        return self.__get_time_string(std) if as_string else std

    def get_total_etc(self,as_string=False):
        t_etc = self.max_counts*self.get_avg()
        return self.__get_time_string(t_etc) if as_string else t_etc

    def get_etc(self,as_string=False):
        etc=(self.max_counts-len(self.deltas))*self.get_avg()
        return self.__get_time_string(etc) if as_string else etc

    def get_stats(self):
        t_stats = {
            'PID': os.getpid(),
            'Elapsed_Time'    : self.__get_time_string(np.sum(self.deltas)),
            'Average_Time'    : f"{self.get_avg(True)} \u00b1 {self.get_std(True)}",
            'Remaining_Time'  : self.get_etc(True),
            'Iteration'       : len(self.deltas),
            'Max_Iterations'  : self.max_counts,
            'Percent_Complete': f"{len(self.deltas)*100.0/self.max_counts:6.2f}%",
        }
        return t_stats 



























