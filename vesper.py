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

import os
import subprocess

# FIXME: This doesn't detect changes to the code.
if not os.path.isfile("parPinger_wrapper.cpp"): # has not yet been compiled by cython
    print("Compiling required Cython libraries")
    cmd = "python setup.py build_ext --inplace"
    subprocess.call(cmd,shell=True)
    print("Done.")

import parPinger_wrapper as pp # an extrernal c++ library which performs all the fast-paced probing activities
import time
import pickle as pkl
import numpy as np
import sys
from prettytable import PrettyTable
from collections import OrderedDict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import datetime
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import logging

'''
Run this script from the command line with -h to see instructions and arguments

#> python vesper.py -h
'''

class Monitor:
    def __init__(self, profiles_path="vesper_profiles/", target_ips=[""], num_trainprobes=-1, probe_interval=-1, rt_plotting=False, window_size = 20, delete_cache=False, log="none"):
        # Initialize the logger
        self.log = logging.Logger(__name__)
        self.log.setLevel(logging.INFO)
        if log == "none":
            self.log.disabled = True
        elif log == "stderr":
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                fmt="[%(asctime)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%zZ")) # ISO 8601 format with timezone offset
            self.log.addHandler(handler)
        else:
            raise Exception(f"Unknown logging backend '{log}'")

        #Try to load last used configuration
        retrain = False
        self.initialized = False
        self.config = None
        if os.path.isfile("config.pkl"):
            self.config = self.load_obj("config")
            # take the given arguments if they are not the default

        if delete_cache:
            if self.config is not None:
                prof_path = self.config['profiles_path']
            else:
                prof_path = "vesper_profiles/"
            if os.path.isdir(prof_path):
                try:
                    for file in os.listdir(prof_path):
                        if file.split('.')[-1] == "pkl":
                            os.remove(os.path.join(self.config['profiles_path'],file))
                    os.rmdir(prof_path)
                except:
                    self.log.error("Could not remove profile directory: "+prof_path)
            self.log.info("All configurations and profiles have been cleared.")

            if self.config is not None:
                os.remove('config.pkl')
                self.config = None

        if self.config is not None:
            if profiles_path != self.config['profiles_path']:
                self.config['profiles_path'] = profiles_path
            if target_ips[0] != "":
                self.config['target_ips'] = target_ips
            if (num_trainprobes != self.config['num_trainprobes']) and (num_trainprobes != -1):
                self.config['num_trainprobes'] = num_trainprobes
                retrain = True

            if probe_interval != -1:
                self.config['probe_interval'] = probe_interval
            if rt_plotting != self.config['rt_plotting']:
                self.config['rt_plotting'] = rt_plotting
            if window_size != self.config['window_size']:
                self.config['window_size'] = window_size
        else: #new config
            self.config = {}
            self.config['profiles_path'] = profiles_path
            self.config['target_ips'] = target_ips
            if num_trainprobes == -1:
                self.config['num_trainprobes'] = 200
            else:
                self.config['num_trainprobes'] = num_trainprobes
            if probe_interval == -1:
                self.config['probe_interval'] = 0
            else:
                self.config['probe_interval'] = probe_interval
            self.config['rt_plotting'] = rt_plotting
            self.config['window_size'] = window_size

        #Load list of target IPs (if exists)
        self.targetIPs = []
        if self.config['target_ips'][0][-4:] == ".csv":
            ipfile = self.config['target_ips'][0]
            if os.path.isfile(ipfile):
                file = open(ipfile, 'r')
                ips = file.readlines()
                file.close()
                if len(ips) == 0:
                    raise Exception(ipfile +" is empty. Type 'vesper -h' for help")
                self.targetIPs = [ip.rstrip() for ip in ips]
            else:
                raise Exception("Can't find "+ipfile+". Type 'vesper -h' for help")
        elif self.config['target_ips'][0] != "":
            for ip in self.config['target_ips']:
                self.targetIPs.append(ip)
        elif delete_cache:
            return
        else:
            raise Exception("No target IP(s) were provided. Type 'vesper -h' for help")

        #Load profiles (if exist)
        if not os.path.isdir(self.config['profiles_path']):
            os.mkdir(self.config['profiles_path'])
        profs = os.listdir(self.config['profiles_path'])
        self.profiles = {}
        for ip in self.targetIPs:
            for prof in profs:
                if ip == prof[:-4]:
                    try:
                        self.profiles[ip] = self.load_obj(os.path.join(self.config['profiles_path'], ip))
                    except EOFError as e:
                        self.log.warning(f"Warning: Couldn't read the profile for {ip}. Retraining instead.")
                        self.log.exception(e)
                        self.profiles[ip] = Profile(ip, self.config['num_trainprobes'], score_window=self.config['window_size'])
                    if retrain:
                        self.profiles[ip].set_train_size(self.config['num_trainprobes'])
                    if self.profiles[ip].score_window != self.config['window_size']:
                        self.profiles[ip]._last_scores.set_size(self.config['window_size'])
                        self.profiles[ip]._last_labels.set_size(self.config['window_size'])
                    break
            if not ip in self.profiles: #did not find profile for the given ip
                self.profiles[ip] = Profile(ip, self.config['num_trainprobes'], score_window=self.config['window_size'])

        # Check for root privileges
        if os.geteuid() != 0:
            self.log.error("This script needs to be run with sudo, to open raw sockets.")
            return

        #save current config to disk
        self.save_obj(self.config,"config")

        #Init plot
        if self.config['rt_plotting']:
            fig = plt.figure()
            self.axis = fig.add_subplot(111)

        # Init parallel prober
        print("Loading Prober")
        self.start_time = time.time()
        self.prober = pp.PyParPinger()
        self.establish_ping_intervals()
        self.initialized = True

    def establish_ping_intervals(self):
        # Establish ping intervals for each IP
        print("Establishing ping transmission frequencies...")
        i=0
        for ip, profile in self.profiles.items():
            if profile.tx_interval == -1:
                print("Sampling: " + profile.ip_addr)
                self.prober.set_target_ip(bytes(profile.ip_addr,"ascii"))
                
                profile.set_tx_interval(self.prober.get_interval())
                if (profile.tx_interval > 0.001) or (profile.tx_interval < 0):
                    print(profile.ip_addr + " took too long to respond. Using 1Khz.")
                    print("Is "+profile.ip_addr+" inside your LAN?")
            else: #use saved value
                print("Loading: " + profile.ip_addr)
            i+=1

    def save_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        """
        Deserialize the saved profile.
        """
        with open(name + '.pkl', 'rb') as f:
            return pkl.load(f)
            
    def run(self):
        if len(self.targetIPs) == 0:
            raise Exception('Cannot run prober if no target IPs have been set.')
        if self.config['rt_plotting']:
            self.plot_score_setup()

        self.start_time = time.time()
        probe_count = 0
        while True:
            # Random order IPs to be probed
            order = np.random.permutation(len(self.targetIPs))
            status = OrderedDict()
            
            for indx in order:
                #prep prober
                targetIP = self.targetIPs[indx]
                profile = self.profiles[targetIP]
                
                self.prober.set_target_ip(bytes(targetIP,"ascii"))
                self.prober.set_ping_interval_sec(profile.tx_interval)
                
                #probe IP
                start = time.time()
                raw_probe = self.prober.probe()
                stop = time.time()
                probe_count += 1

                #execute/train profile
                
                label, score = profile.process(raw_probe)
                
                status[targetIP] = [label,score,profile.trainProgress(),profile.tx_interval,stop-start,profile.n_packets_lost_lastprobe]

                time.sleep(self.config['probe_interval']/1000)

            # report
            self.report(status,probe_count)

            # plot
            if self.config['rt_plotting']:
                self.plot_score_update()

            # save updated profiles
            if int(probe_count/len(self.targetIPs)) % 50 == 0:
                self.saveProfiles()

            status.clear()

    def saveProfiles(self):
        print("Saving Profiles...")
        for ip, profile in self.profiles.items():
            if profile._updated:
                self.save_obj(profile, os.path.join(self.config['profiles_path'], ip))


    def report(self, status, probe_count):
        table = PrettyTable()
        table.field_names = ['IP','Status','Score','Profile','Tx Freq [kH]','Probe Duration', 'Note']
        table.sortby = "IP"
        for ip, status in status.items():
            label = status[0]
            score = np.round(status[1],2)
            trainProgress = status[2]
            tx_interval = status[3]
            probe_time = status[4]
            lost_packets = status[5]
            prof = str(np.round(trainProgress*100,1))+'%'
            state = 'Normal'
            note = ''
            if label == -1:
                state = 'Abnormal'
                note = 'Abnormal connection detected'
                self.alert(ip)
            if label == -2:
                if score >= 0:
                    state = 'Normal?'
                    note = 'Losing packets ('+str(lost_packets)+'/1023)'
                else:
                    state = 'Abnormal'
                    note = 'Abnormal connection detected. Losing Packets ('+str(lost_packets)+'/1023)'
                    self.alert(ip)
            if label == -3:
                state = 'unknown'
                note = 'Lost connection'
            if trainProgress == 1:
                prof = 'Trained'
            else:
                state = '-'
                score = '-'
                note = 'In training'
            table.add_row([ip, state, score, prof,np.round((1/tx_interval)/1000,2), str(np.round(probe_time*1000,2)) +' ms', note])
        os.system('cls' if os.name=='nt' else 'clear')
        print("Vesper Status  --  Runtime: "+ "{:0>8}".format(str(datetime.timedelta(seconds=time.time()-self.start_time))))
        print(table)
        print("Sent "+"{:,}".format(probe_count)+" probes.")

    def alert(self, ip):
        self.log.warning(f"Security Alert: Vesper detected a possible Man-in-the-Middle-Attack to {str(ip)}. The connection latency profile has changed. This could also be caused by an intended change in network topology.")

    def plot_score_setup(self):
        curTime_min = (time.time() - self.start_time) / 60
        for ip, profile in self.profiles.items():
            y = profile._last_labels.get_mean()
            self.axis.plot(curTime_min, y, label = ip)
        plt.legend(loc='upper left')
        plt.xlabel('Time Elapsed [min]')
        plt.ylabel('Anomaly Score')
        self.axis.axhline(y=0,linestyle='--',color='red')
        plt.ion()
        plt.show(block=False)

    def plot_score_update(self):
        if plt.get_fignums(): # the plot window is open
            plt.ion()
            curTime_min = (time.time()-self.start_time)/60
            for line in self.axis.get_lines():
                #find profile
                profile = self.profiles.get(line.get_label())
                if profile is not None:
                    x = line.get_xdata()
                    y = line.get_ydata()
                    if len(x) >= 200:
                        line.set_xdata(np.concatenate((x[1:],[curTime_min])))
                        line.set_ydata(np.concatenate((y[1:],[profile._last_labels.get_mean()])))
                    else:
                        line.set_xdata(np.concatenate((x,[curTime_min])))
                        line.set_ydata(np.concatenate((y,[profile._last_labels.get_mean()])))
            #self.axis.relim()
            #self.axis.autoscale_view()
            self.axis.set_xlim(self.axis.get_lines()[0].get_xdata().min(),curTime_min)
            self.axis.set_ylim(-1.1,1.1)
            plt.ioff()
            plt.draw()
            plt.pause(0.01)
            plt.show(block=False)


class Profile:
    def __init__(self,ip,train_size=100,tx_interval=-1,score_window=10):
        self.ip_addr = ip
        self.tx_interval = tx_interval
        self.train_size = train_size
        self.samples = []
        self.detector = LocalOutlierFactor(novelty=True)
        self.scaler = StandardScaler()
        self.KS_population = []
        self._updated = False
        self._last_vjits = ringwindow(15)
        self.score_window = score_window # the averaging window used over the anomaly scores. Larger windows increase robustness bu increase detection delay too.
        self._last_scores = ringwindow(self.score_window,1)
        self._last_labels = ringwindow(self.score_window,1)
        self.n_packets_lost_lastprobe = 0

    def set_ip(self,ip):
        self.ip_addr = ip
        self._updated = True

    def set_tx_interval(self,value):
        self.tx_interval = value
        self._updated = True

    def set_train_size(self,value):
        self.train_size = value
        self.samples = self.samples[np.max((len(self.samples) - self.train_size, 0)):]  # take top most recent samples
        if not self.inTraining(): #refit model to current samples
            self.scaler.fit(np.vstack(self.samples))
            self.detector = self.detector.fit(self.scaler.transform(np.vstack(self.samples)))
        self._updated = True

    def trainProgress(self):
        return np.double(len(self.samples))/np.double(self.train_size)

    def inTraining(self):
        return len(self.samples) < self.train_size

    def process(self, raw_probe):
        """
        raw_probe is the probe data generated by parPinger::probe().
        Returns the label and score.
        """
        
        #check probe integrity
        n_lost_packets = np.sum(np.array(raw_probe[1])==0) #number of those with no response
        if n_lost_packets == len(raw_probe[1]): #all packets were lost
            return -3, self._last_labels.get_mean()
        if n_lost_packets > 0: #some packets were lost (we can't accuralty compute the probe)
            self.n_packets_lost_lastprobe = n_lost_packets
            if n_lost_packets <= 200: #we will still try and execute if only a few were lost
                # perform partial feature extraction
                sample = self.extract_features_partial(raw_probe)

                # execute partial profile
                return self._process(sample, wasPartial=True)
            else:
                return -2, self._last_labels.get_mean()
        else: #no packets lost:
            self.n_packets_lost_lastprobe = 0

            #perform feature extraction
            sample = self.extract_features(raw_probe)

            #train/execute profile
            return self._process(sample)

    def _process(self, x, wasPartial=False):
        """
        Learns and then scores sample. 
        When still in training, special values are returned.
        """

        if self.inTraining() and wasPartial:
            return -2, 1
        if self.inTraining():
            self.samples.append(x)
            self.samples = self.samples[np.max((len(self.samples)-self.train_size,0)):] #take top most recent samples
            if not self.inTraining():
                self.scaler.fit(np.vstack(self.samples))
                self.detector = self.detector.fit(self.scaler.transform(np.vstack(self.samples)))
            self._updated = True
            return 1, 1.0
        else:
            if wasPartial:
                label = self.classify_sample(x) #update scores
                label = -2
            else:
                label = self.classify_sample(x)
            score = self._last_labels.get_mean()
            return label, score

    def score_sample(self,x):
        if self.inTraining():
            return #1.0
        else: #model is trained
            return self._last_scores.insert_get_mean(self.detector.decision_function(self.scaler.transform(x))[0])# * -1  # larger is more anomalous

    def classify_sample(self,x):
        if self.inTraining():
            return 1
        else: #model is trained
            m_label = self._last_labels.insert_get_mean(self.detector.predict(self.scaler.transform(x))[0]) #1:normal, -1:anomaly
            return -1 if m_label < 0 else 1

    def extract_features(self,raw_probe):
        """
        What these features mean is discussed in `4.4. Feature Extractor (FE)` of the paper.
        """

        tx_times = np.array(raw_probe[0])
        rx_times = np.array(raw_probe[1])
        mls_seq = np.array(raw_probe[2])

        # Feature 1: v_ie
        # Total energy of impulse
        rtt = rx_times - tx_times
        rtt_f = np.fft.fft(rtt)
        mls_seq_f = np.fft.fft(mls_seq)
        v_ie = np.sum(np.abs((rtt_f / mls_seq_f)) ** 2) / len(rtt_f)

        # Feature 2: v_dc
        # The mean round trip time of the largest payload pings
        if (mls_seq == 0).all():  
            # should not happen (means MLS was all zeros)
            v_dc = np.mean(rtt)
        else:
            v_dc = np.mean(rtt[mls_seq == 1])

        # Feature 3: v_jit
        # Log-likelihood of the Jitter’s Distribution. (But capped to 0 or 1).
        jitter = np.diff(rx_times, n=1)
        if len(self.KS_population) == 0:
            m_pv = 1
        else:
            pvs = np.zeros(len(self.KS_population))
            for i in range(len(self.KS_population)):
                pvs[i] = ks_2samp(self.KS_population[i], jitter)[0]
            m_pv = np.max(pvs)
        v_jit = 0.0 if m_pv < 0.1 else 1.0

        # update KS model
        set_size=30
        if self.inTraining():
            if (len(self.KS_population) < set_size) or (np.random.rand() > 0.7):
                self.KS_population.append(jitter)
                self.KS_population = self.KS_population[np.max((len(self.KS_population)-set_size,0)):]
                self._updated = True
        
        # FIXME: Make this a class. This ad-hoc definition is used in several places.
        return np.array([[v_ie, v_dc, v_jit]])

    def extract_features_partial(self,raw_probe):
        tx_times = np.array(raw_probe[0])
        rx_times = np.array(raw_probe[1])
        mls_seq = np.array(raw_probe[2])
        good = rx_times != 0
        rtt = rx_times[good] - tx_times[good]
        average_sample = np.mean(np.vstack(self.samples),axis=0)

        # Feature 1: v_ie AVERAGE (not tested)
        v_ie = average_sample[0]

        # Feature 2: v_dc
        if (mls_seq == 0).all():  # should not happen (means MLS was all zeros)
            v_dc = np.mean(rtt)
        else:
            v_dc = np.mean(rtt[mls_seq[good] == 1])  # the average rtt of the largest payload pings

        # Feature 3: v_jit AVERAGE (not tested)
        v_jit = average_sample[2]

        return np.array([[v_ie, v_dc, v_jit]])


class ringwindow:
    def __init__(self, winsize, fill=np.nan):
        self.buffer = np.zeros(int(winsize))
        self.next_i = 0
        self.haslapsed = False
        self.fill = fill
        if fill != np.nan:
            self.buffer += fill
            self.haslapsed = True

    def insert(self,v):
        self.buffer[self.next_i] = v
        self.next_i += 1
        if self.next_i == len(self.buffer):
            self.haslapsed = True
        self.next_i = np.mod(self.next_i,len(self.buffer))

    def get_mean(self):
        if self.haslapsed:
            return np.mean(self.buffer)
        return np.mean(self.buffer[:self.next_i])

    def insert_get_mean(self,v):
        self.insert(v)
        return self.get_mean()

    def set_size(self,w):
        w = int(w)
        if w > len(self.buffer):
            self.next_i = len(self.buffer)
            B = np.ones(int(w))*self.fill
            B[:len(self.buffer)] = self.buffer
            self.buffer = B
        if w <= len(self.buffer):
            self.next_i = np.mod(self.next_i,w)
            self.buffer = self.buffer[:w]







if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.description = description='VESPER: A man-in-the-middle detector for LAN environments. By Yisroel Mirsky 2019\nBounces ICMP packet ' \
                                     'singals off target hosts and anylizes the response in order to determine if the ' \
                                     'environment (link) has changed...\n\n        ...like a bat in the dark  :).\n\nCan be used to ' \
                                     'detect when additional devices have been added to a link or when existing devices ' \
                                     '(e.g., switches) have been swapped with different ones.'
    parser.epilog = 'When launched, vesper will monitor the IPv4 addresses in the local file IPs.csv, unless a target IP ' \
                    'address is provided as an argument. A profile is trained for each host and is saved to disk ' \
                    '(automatically retrieved each time the tool is started). The configuration of the last run is saved to disk (except the realtime plotting toggle argument). Note, this tool only works when monitoring ' \
                    'a link contained within a LAN (switches only). Do not provide external IPs.\n\n'\
                    'For more information, please read our paper:\nVesper: Using Echo-Analysis to Detect Man-in-the-Middle ' \
                    'Attacks in LANs\nYisroel Mirsky, Naor Kalbo, Yuval Elovici, Asaf Shabtai'
    parser.add_argument('-i',default=[""],nargs='*',help="Monitor the given IP address(es) <I> only. If an IP's profile exists on disk, it will be loaded and used.\nYou can also provide the path to a file containing a list of IPs, where each entry is on a seperate line.\nExample: python vesper.py -i 192.168.0.12\nExample: python vesper.py -i 192.168.0.10 192.168.0.42\nExample: python vesper.py -i ips.csv")
    parser.add_argument('-t',type=int,default=200,help="set the train size with the given number of probes <T>. If profiles already exist, the training will be shortened or expanded accordingly. a\nDefault is 200.\nExample: python vesper.py -i 192.168.0.12 -t 400")
    parser.add_argument('-p',action='store_true',help="Plot anomaly scores in realtime. \nExample: python vesper.py -p")
    parser.add_argument('-f',default="vesper_profiles/",help="load/save profiles from the given directory <F>. If is does not exist, it will be created. \nDefault path is ./vesper_profiles.")
    parser.add_argument('-r',type=int,default=0,help="Sets the wait time <R> between each probe in miliseconds. \nDefault is 0.")
    parser.add_argument('-w',type=int,default=10,help="Sets the sliding window size <W> used to average the anomaly scores. A larger window will provide fewer false alarms, but it will also increase the detection delay. \nDefault is 10.")
    parser.add_argument('--reset',action='store_true',help="Deletes the current configuration and all IP profiles stored on disk before initilizing vesper")
    parser.add_argument('--log', default="none", choices=["none", "stderr"], type=str, help='Where to log alerts. Possible values are "none" and "stderr". ')

    args = parser.parse_args()

    make_plot = True if (args.p is not None) else False
    mon = Monitor(profiles_path=args.f, target_ips=args.i, num_trainprobes=args.t, probe_interval=args.r, rt_plotting=args.p, window_size=args.w, delete_cache = args.reset, log=args.log)
    if mon.initialized: # False if there was an error or termination during initilization
        mon.run()
