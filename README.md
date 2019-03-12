# Overview
In this repository you will find a man in the middle detection tool for local area networks. 

This tool was presented at **Black Hat Asia 2019** ([here](https://www.blackhat.com/asia-19/briefings/schedule/index.html#see-like-a-bat-using-echo-analysis-to-detect-man-in-the-middle-attacks-in-lans-13713))

## What is Vesper?

Although Man-in-the-Middle (MitM) attacks on LANs have been known for some time, they are still considered a significant threat. This is because these attacks are relatively easy to achieve, yet challenging to detect. For example, a planted network bridge or compromised switch leaves no forensic evidence.

![alt text](https://raw.githubusercontent.com/ymirsky/Vesper/master/imgs/vesper_logo.png)
 is a novel plug-and-play MitM detector for local area networks. Vesper uses a technique inspired from the domain of acoustic signal processing. Analogous to how echoes in a cave capture the shape and construction of the environment, so to can a short and intense pulse of ICMP echo requests model the link between two network hosts. Vesper sends these probes to a target network host and then uses the reflected signal to summarize the channel environment (think sonar). Vesper uses machine learning to profile the link with each host, and to detect when the environment changes. Using this technique, Vesper can detect MitM attacks with high accuracy, to the extent that it can distinguish between identical networking devices. 

![An illustration of Vesper's architecture](https://raw.githubusercontent.com/ymirsky/Vesper/master/imgs/framework.png)

Vesper has been evaluated on LANs consisting of video surveillance cameras, servers, and hundreds of PC workstations. The tool also works across multiple network switches and in the presence of traffic. Although Vesper is robust against adversarial attacks, the current version of the tool does not implement some of the defensive measures.

## What can Vesper Do?

There are many kinds of MitM attacks when it comes to LANs. We categorize the class of a MitM attack based on the MitM topology, and implementation used. The figures below shows the attack topologies, implementations, and notes how well Vesper can detect them. 

![An illustration of Vesper's architecture](https://raw.githubusercontent.com/ymirsky/Vesper/master/imgs/types_of_mitm.png)


## How Does Vesper Work (brief)
A MitM will always affect packet timings for two reasons:
1. to avoid signal collisions on the media when transmitting crafted/altered packets, and 
2. to capture and alter relevant packets before they reach their intended destination. In the latter case, the MitM must parse every frame in order to determine the frame's relevancy to the attack, and cannot retroactively stop a transmitted frame.

Therefore, the interception process (hardware and/or software) will affect the timing of network traffic. We note that since passive wiretaps only observe traffic, they are not MitM attacks and therefore not in the scope of this paper. However, Vesper can detect a MitM which is presently eavesdropping (not currently altering traffic) because a MitM always buffers each packet upon reception. The figure below illustrates the basic packet interception process for all MitM implementations. 

![An illustration of Vesper's architecture](https://raw.githubusercontent.com/ymirsky/Vesper/master/imgs/interception.png)

However, measuring timing alone is not enough since two similar devices will have the same timing. Instead vesper tries to fingerprint the devices along the link by modeling the environment's impulse response. 

In networks, there are no reverberations of sound waves. However, switches, network interfaces, and operating systems all affect a packet's travel time across a network. The hardware, buffers, caches, and even the software versions of the devices which interact with the packets, all affect packet timing. When a device processes a burst of packets, the device has dynamic reaction with respect to the packets' sizes. This affects the packets' processing times, which are in turn, then propagated to the next node in the network. This is analogous to how a sound wave is affected as it reverberates off various surfaces.

Vesper monitors changes to the 'environment' (link) by sending periodic probes to a target network host. The probe is a special burst of ICMP packets (pings) whose sizes are modulated according to an MLS signal. MLS is used to make the response robust to noise and strong against adversarial attacks (on Vesper). The response signal which bounces back to Vesper is then used to extract three summary features that measure the impulse response energy, the dominant DC component, and the packet jitter distribution. Using these probes, Vesper profiles each link with an anomaly detection algorithm.


## Where is Vesper Deployed?

In order to protect the link between hosts A and B, Vesper only needs to be running on host A. However, this trust is one-sided since B would be unaware of the state of his link with A. Therefore, to secure all links in a LAN in a fully trusted manner, all hosts in the LAN must be running an instance of Vesper. This kind of deployment can be practical in large LANs if Vesper is configured to send probes at a low rate or only while communicating with the target end-host.

Another option is to install an instance of Vesper on the network gateway (router). Although this does not secure the links between each host of the LAN, it does secure the inbound and outbound traffic. Note that both deployments only protect a host from MitM attacks originating within the same LAN. 

![An illustration of Vesper's architecture](https://raw.githubusercontent.com/ymirsky/Vesper/master/imgs/usages.png)


# The Code

## Implementation Notes: 

* This is a python implementation of Vesper which wraps C/C++ code using cython. The C/C++ code is used to perform the ICMP probing quickly and accurately.
* This implementation uses local outlier factor (LOF) for anomaly detection (BlackHat'19) and not autoencoders (NDSS'18).
* The current version of vesper has been tuned to detect all of these cases except IP-DH where the exact same model is being used (i.e., the tool can detect the difference between two different 1Gps switches, but not identical ones). The tuned version will be released at a later date.  
* This tool currently does not currently implement detection of attacks on Vesper itself. 
* The source code has been tested with Python 2.7.12 on a Linux 64bit machine (Kali). To port the tool to Windows, some C++ libraries must be changed.
* Python dependencies: prettytable, cython  

To install prettytable and cython, run this in the terminal:
```
pip install prettytable cython
```
 


## Using the Tool
Since the tool uses raw sockets, you **must** run vesper with sudo privileges. For example:
```
$ sudo python vesper.py
```

The first time you run vesper.py, cython will compile the necessary C++ libraries. When launched, Vesper will monitor the IPv4 addresses in the local file IPs.csv, unless a target IP address is provided as an argument. A profile is trained for each host and is saved to disk (automatically retrieved each time the tool is started). The configuration of the last run is saved to disk (except the real-time plotting toggle argument). Note, this tool only works when monitoring a link contained within a LAN (switches only). Do not provide external IPs.

For complete instructions on how to use vesper, type into the command line
```
$ python vesper.py -h

usage: vesper.py [-h] [-i [I [I ...]]] [-t T] [-p] [-f F] [-r R] [-w W]
                 [--reset]

optional arguments:
  -h, --help      show this help message and exit
  -i [I [I ...]]  Monitor the given IP address(es) <I> only. If an IP's profile exists on disk, it will be loaded and used.
                  You can also provide the path to a file containing a list of IPs, where each entry is on a separate line.
                  Example: python vesper.py -i 192.168.0.12
                  Example: python vesper.py -i 192.168.0.10 192.168.0.42
                  Example: python vesper.py -i ips.csv
  -t T            set the train size with the given number of probes <T>. If profiles already exist, the training will be shortened or expanded accordingly. a
                  Default is 200.
                  Example: python vesper.py -i 192.168.0.12 -t 400
  -p              Plot anomaly scores in real-time. 
                  Example: python vesper.py -p
  -f F            load/save profiles from the given directory <F>. If is does not exist, it will be created. 
                  Default path is ./vesper_profiles.
  -r R            Sets the wait time <R> between each probe in milliseconds. 
                  Default is 0.
  -w W            Sets the sliding window size <W> used to average the anomaly scores. A larger window will provide fewer false alarms, but it will also increase the detection delay. 
                  Default is 10.
  --reset         Deletes the current configuration and all IP profiles stored on disk before initializing vesper.
```

# License
See the [LICENSE](LICENSE) file for details


# Citations
If you use the source code in any way, please cite our paper:

*Mirsky Y, Kalbo N, Elovici Y, Shabtai A. Vesper: Using Echo Analysis to Detect Man-in-the-Middle Attacks in LANs. IEEE Transactions on Information Forensics and Security. 2019 Jun;14(6):1638-53.*

Yisroel Mirsky
yisroel@post.bgu.ac.il
