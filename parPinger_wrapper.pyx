
# MIT License
#
# Copyright (c) 2019 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The software is not to be sold or used for non-commerical purposes, and the above copyright notice
# and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


#This is a Cython file and extracts the relevant classes from the C++ header file.

# distutils: language = c++
# distutils: sources = parPinger.cpp

import numpy as np
from libcpp.vector cimport vector
ctypedef unsigned short uns16
ctypedef long double ld

cdef extern from "parPinger.hpp" namespace "pinger":
    cdef cppclass parPinger:
        parPinger(char*,double, uns16)
        vector[vector[ld]] probe()
        double get_interval()
        void set_ping_interval_sec(double)
        void set_target_ip(char *, uns16)

cdef class PyParPinger:
    cdef parPinger *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, char* ip = "", double ping_interval_sec = -1):
        self.thisptr = new parPinger(ip, ping_interval_sec, np.uint16(hash(ip)))
    def __dealloc__(self):
        del self.thisptr
    def probe(self):
        return self.thisptr.probe()
    def get_interval(self):
        return self.thisptr.get_interval()
    def set_ping_interval_sec(self, double value):
        self.thisptr.set_ping_interval_sec(value)
    def set_target_ip(self, char* ip):
        self.thisptr.set_target_ip(ip,np.uint16(hash(ip)))