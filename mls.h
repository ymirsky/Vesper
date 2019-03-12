/*
# MIT License
#
# Copyright (c) 2019 Yisroel Mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The software is to be used for non-commerical purposes, not to be sold, and the above copyright notice
# and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
*/

#ifndef MLS_H
#define MLS_H

#include <vector>
#include <cmath>
#include <time.h>
#include <unordered_map>
#include <stdlib.h>
#include <random>
#include "aes.hpp"

using namespace std;

class mls
{
public:
    mls(int nbits = 10, bool useAES = true);
    vector<bool> get_seq();
    void setBits(int n_bits);
    int size();

private:
    unordered_map<int, vector<int>> Taps;
    int nbits;
    bool isAES;
    random_device rd; //used for generating non-deterministic random numbers
    bool goodSeq(vector<bool> seq);

    //Used in the case of AES
    uint8_t *key; //initialized with random device
    uint8_t *iv;
    struct AES_ctx ctx;

};

#endif // MLS_H
