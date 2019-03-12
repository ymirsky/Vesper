#include "parPinger.hpp"
//#include "parpinger.h"
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/syscall.h>
#include <math.h>
#include <algorithm>
#include <limits>
#include <iomanip>

namespace pinger
{
  parPinger::parPinger(char* ip, double ping_interval_sec, uint16_t threadID)
  {
        //parameters
        this->targetIP = ip;
        this->threadID = threadID;
        if(ping_interval_sec < 0)
            ping_interval_sec = 0.0001;
        this->set_ping_interval_sec(ping_interval_sec);
        this->pir_count = 0;
        this->MLSgen.setBits(10); //1023 length
  }
  parPinger::~parPinger() {}

  void parPinger::set_target_ip(char* ip, uint16_t threadID){
        this->targetIP = ip;
        this->threadID = threadID;
  }


  void parPinger::set_ping_interval_sec(double value){
        this->ping_interval.tv_sec = (long) (((long double)value)*0.58);
        this->ping_interval.tv_nsec = (((long double)value)*0.58 - ping_interval.tv_sec)* 1000000000L;
  }

  std::vector<std::vector<long double>> parPinger::probe()
  {
    //Clear data from last probe
    currPIR_MLS.clear();
    send_times.clear();
    curResult.clear();

    if(this->targetIP.compare("") == true){
        cout<<"parProber: Cannot probe without first setting an IP!"<<endl;
        return curResult;
    }

    //Make threads
    pthread_create(&probeRecverThread, NULL, parPinger::recvMain, (void*)this);
    pthread_create(&probeSenderThread, NULL, parPinger::sendMain, (void*)this);

    pthread_join(probeRecverThread,NULL);//wait for threads to complete
    return curResult;
  }


//Receives a burst of ICMP ECHO REPLIES, and measures their RTT times.
void* parPinger::recvMain(void *args)
{
    parPinger* prthr = (parPinger*)args;
    struct timespec receptionTime;
    vector<point> recv_times; //temporary place to store the points from a single PIR
    point rx;
    clock_gettime(CLOCK_MONOTONIC, &rx.t);

    /*Setup receive socket*/
    int s, i, cc, packlen, datalen = 1500 + ICMP_MINLEN;
    struct sockaddr_in to, from;
    fd_set rfds;
    int ret, fromlen, hlen;
    int retval;
    struct icmp *icp;
    to.sin_family = AF_INET;
    string hostname;
    u_char *packet, outpack[MAXPACKET];
    struct ip *ip;

    // try to convert as dotted decimal address, else if that fails assume it's a hostname
    to.sin_addr.s_addr = inet_addr(prthr->targetIP.c_str());
    if (to.sin_addr.s_addr != (u_int)-1)
        hostname = prthr->targetIP;
    else
    {
        cerr << "unknown host "<< prthr->targetIP << endl;
        return NULL;
    }
    packlen = datalen + MAXIPLEN + MAXICMPLEN;
    if ( (packet = (u_char *)malloc((u_int)packlen)) == NULL)
    {
        cerr << "malloc error\n";
        return NULL;
    }
    if ( (s = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0)
    {
        perror("socket");	/* probably not running as superuser */
        free(packet);
        return NULL;
    }
    // Watch for socket inputs.
    FD_ZERO(&rfds);
    FD_SET(s, &rfds);

    // Timeout: after each reception, wait up to X micro seconds before giving up on remaining replies.
    timeval timeout;
    struct timespec curTime;

    /* Main loop */
    int rx_count = 0;
    for(;;)
    {
        /* Receive ICMP ECHO Response */
        timeout.tv_sec = 2;
        timeout.tv_usec = 0;

        retval = select(s+1, &rfds, NULL, NULL, &timeout);
        clock_gettime(CLOCK_MONOTONIC, &receptionTime); //reception time
        if (retval == -1)
        {
            perror("select()");
            break;
        }
        if (retval == 0)//timeout
        {
            break;
        }
        fromlen = sizeof(sockaddr_in);
        if ( (ret = recvfrom(s, (char *)packet, packlen, 0,(struct sockaddr *)&from, (socklen_t*)&fromlen)) < 0)
        {
            perror("recvfrom error");
            //break;
        }

        // Check the IP header
        ip = (struct ip *)((char*)packet);
        hlen = sizeof( struct ip );
        if (ret < (hlen + ICMP_MINLEN))
        {
            cout<<"ECHO receive Error"<<endl;
            //break;
        }

        // Now the ICMP part
        icp = (struct icmp *)(packet + hlen);
        if (icp->icmp_type == ICMP_ECHOREPLY)
        {
            //cout << "Recv: echo reply"<< endl;
            if (icp->icmp_id != prthr->threadID)
                continue;
        }
        else
        {// cout << "Recv: not an echo reply" << endl;
            continue;
        }

        /* Capture the response */
        rx.t = receptionTime;
        rx.indx = icp->icmp_seq;
        recv_times.push_back(rx);
        rx_count++;

        if(rx_count == prthr->currPIR_MLS.size())
            break;
    }

    /*Clean up*/
    close(s);
    free(packet);


    //if(recv_times.size()!=0){
    /////// Compute RTTs ///////
    vector<long> indexes(prthr->send_times.size(),0);
    vector<double> tx_time(prthr->send_times.size(),0);
    vector<double> rx_time(prthr->send_times.size(),0);
    vector<double> rtt(prthr->send_times.size(),0);
    vector<timespec> rtt_ts(prthr->send_times.size());
    vector<timespec> rx_ts(prthr->send_times.size());

    int rx_indx = 0;
    struct timespec empty_ts;
    long indx_cntr = 0;
    bool missing = false;
    int misscount=0;
    for(int i=0;i<prthr->send_times.size();i++)//assume that replies are in same order as requests
    {
        indx_cntr++;
        indexes[i] = indx_cntr;
        tx_time[i] = (double)(prthr->send_times[i].t.tv_sec) + ((double)(prthr->send_times[i].t.tv_nsec)/1000000000.0);
        if(rx_indx < recv_times.size())//has more received entries to process
        {
            if(prthr->send_times[i].indx==recv_times[rx_indx].indx){
                rx_time[i] = (double)(recv_times[rx_indx].t.tv_sec) + ((double)(recv_times[rx_indx].t.tv_nsec)/1000000000.0);
                rtt[i] = rx_time[i] - tx_time[i];
                if(rtt[i]<=0)//rounding float error
                {
                    rtt[i] = ((double)(prthr->tsSubtract(recv_times[rx_indx].t,prthr->send_times[i].t).tv_nsec))/1000000000.0;
                }
                rtt_ts[i] = prthr->tsSubtract(recv_times[rx_indx].t,prthr->send_times[i].t);
                rx_ts[i] = recv_times[rx_indx].t;
                rx_indx++;
            }else{
                missing = true;
                misscount++;
                rx_time[i] = nan("");
                rtt[i] = nan("");
                empty_ts.tv_sec = 0; empty_ts.tv_nsec = 0;
                rtt_ts[i] = empty_ts;
                rx_ts[i] = empty_ts;
            }
        }else{
            missing = true;
            misscount++;
            rx_time[i] = nan("");
            rtt[i] = nan("");
            empty_ts.tv_sec = 0; empty_ts.tv_nsec = 0;
            rtt_ts[i] = empty_ts;
            rx_ts[i] = empty_ts;
        }
    }

    if(missing){
        cout<<"Lost: "<< misscount <<" out of "<< prthr->currPIR_MLS.size() <<" expected responses."<<endl;
    }

    ///// Prep result vectors ////
    vector<long double> TX_TIMES;
    vector<long double> RX_TIMES;
    vector<long double> MLS_SEQ;

    for(int i = 0; i < indexes.size(); i++){
        TX_TIMES.push_back(prthr->ts2ld(prthr->send_times[i].t));
        RX_TIMES.push_back(prthr->ts2ld(rx_ts[i]));
        MLS_SEQ.push_back((long double)prthr->currPIR_MLS[i]);
    }

    vector<vector<long double>> result;
    result.push_back(TX_TIMES);
    result.push_back(RX_TIMES);
    result.push_back(MLS_SEQ);
    prthr->curResult = result;
    recv_times.clear();
}




long double parPinger::ts2ld(struct timespec t)
{
    string ts;
    ts += to_string(t.tv_sec);
    ts += ".";
    string rm = to_string(t.tv_nsec);
    int zeros = 9;
    int nanos = rm.size();
    for(int i =0; i < zeros - nanos; i++)
        ts += "0";
    for(int i =0; i < nanos; i++)
        ts += rm.at(i);
    return stold(ts);
}



struct timespec parPinger::tsSubtract (struct  timespec  time1, struct  timespec  time2)
{    /* Local variables. */
    struct  timespec  result ;

/* Subtract the second time from the first. */

    if ((time1.tv_sec < time2.tv_sec) ||
        ((time1.tv_sec == time2.tv_sec) &&
         (time1.tv_nsec <= time2.tv_nsec))) {		/* TIME1 <= TIME2? */
        result.tv_sec = result.tv_nsec = 0 ;
    } else {						/* TIME1 > TIME2 */
        result.tv_sec = time1.tv_sec - time2.tv_sec ;
        if (time1.tv_nsec < time2.tv_nsec) {
            result.tv_nsec = time1.tv_nsec + 1000000000L - time2.tv_nsec ;
            result.tv_sec-- ;				/* Borrow a second. */
        } else {
            result.tv_nsec = time1.tv_nsec - time2.tv_nsec ;
        }
    }

    return (result) ;
}

  void* parPinger::sendMain(void *args)
  {
    /* Prepare for Probe */
    parPinger* prthr = (parPinger*)args;
    struct timespec curTime;
    prthr->burstTime = 9999999; //init the time it took to send the burst
    usleep(100000); // 100ms

    /* Send Probe */
    //prthr->currPIR_MLS = prthr->MLSgen.get_seq();
    prthr->currPIR_MLS = {1,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,1,0,1,1,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,0,1,1,1,1,0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,1,0,1,1,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,0,0,0,1,1,0,1,0,1,1,0,1,1,1,0,1,0,0,0,1,0,1,0,1,1,1,1,1,1,0,1,0,0,0,1,1,1,0,0,1,1,0,1,1,1,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,1,1,0,1,0,0,1,1,1,1,0,0,1,1,0,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,1,1,1,1,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,1,1,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,1,0,1,0,1,1,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,0,1,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0,1,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,1,1,1,0,1,0,0,1,0,1,0,1,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,0,1,0,0,0,0,1,1,1,0,1,0,0,1,0,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,0,0,1,1,1,1,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,0,0,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,1,1,1};
    prthr->pir_count++;//update id for current pir
    prthr->send_probe(prthr->targetIP, prthr->currPIR_MLS, prthr->threadID);

  }


//gets average RTT in seconds time between this host and target IP
//Each ping is sent AFTER each response. Note: parPinger sends requests back-to-back and receives them in parrallel.
double parPinger::get_interval()
{
    double sumRTTs = 0;
    int s, i, cc, packlen, datalen = 1500;
    struct hostent *hp;
    struct sockaddr_in to, from;
    //struct protoent	*proto;
    struct ip *ip;
    u_char *packet, outpack[MAXPACKET];
    char hnamebuf[MAXHOSTNAMELEN];
    string hostname;
    struct icmp *icp;
    int ret, fromlen, hlen;
    fd_set rfds;
    struct timeval tv;
    int retval;
    struct timespec start, end;

    double /*start_t, */end_t;
    bool cont;

    to.sin_family = AF_INET;

    // try to convert as dotted decimal address, else if that fails assume it's a hostname
    to.sin_addr.s_addr = inet_addr(targetIP.c_str());
    if (to.sin_addr.s_addr != (u_int)-1)
        hostname = targetIP.c_str();
    else
    {
        cerr << "unknown host "<< targetIP << endl;
        return -1;
    }
    packlen = datalen + MAXIPLEN + MAXICMPLEN;
    if ( (packet = (u_char *)malloc((u_int)packlen)) == NULL)
    {
        cerr << "malloc error\n";
        return -1;
    }


    if ( (s = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0)
    {
        perror("socket");	/* probably not running as superuser */
        free(packet);
        return -1;
    }
    pid_t my_tid = syscall(__NR_gettid);

    for(int k = 0; k < 1024; k++){
        icp = (struct icmp *)outpack;
        icp->icmp_type = ICMP_ECHO;
        icp->icmp_code = 0;
        icp->icmp_cksum = 0;
        icp->icmp_seq = k;	/* seq and id must be reflected */
        icp->icmp_id = my_tid % 65000;


        cc = datalen + ICMP_MINLEN;
        icp->icmp_cksum = in_cksum((unsigned short *)icp,cc);

        // Watch stdin (fd 0) to see when it has input.
        FD_ZERO(&rfds);
        FD_SET(s, &rfds);
        // Wait up to X micro seconds.
        tv.tv_sec = 0;
        tv.tv_usec = 200000;

        i = sendto(s, (char *)outpack, cc, 0, (struct sockaddr*)&to, (socklen_t)sizeof(struct sockaddr_in));
        clock_gettime(CLOCK_MONOTONIC, &start); //use CLOCK_MONOTONIC in deployment

        if (i < 0 || i != cc)
        {
            if (i < 0){
                close(s);
                free(packet);
                perror("sendto error");
            }
            cout << "wrote " << hostname << " " <<  cc << " chars, ret= " << i << endl;
        }


        cont = true;
        while(cont)
        {
            retval = select(s+1, &rfds, NULL, NULL, &tv);
            clock_gettime(CLOCK_MONOTONIC, &end); //use CLOCK_MONOTONIC in deployment
            if (retval == -1)
            {
                perror("select()");
                close(s);
                free(packet);
                return 0.001;
            }
            else if (retval)
            {
                fromlen = sizeof(sockaddr_in);
                if ( (ret = recvfrom(s, (char *)packet, packlen, 0,(struct sockaddr *)&from, (socklen_t*)&fromlen)) < 0)
                {
                    perror("recvfrom error");
                    close(s);
                    free(packet);
                    return 0.001;
                }

                // Check the IP header
                ip = (struct ip *)((char*)packet);
                hlen = sizeof( struct ip );
                if (ret < (hlen + ICMP_MINLEN))
                {
                    //   cerr << "packet too short (" << ret  << " bytes) from " << hostname << endl;
                    close(s);
                    free(packet);
                    return 0.001;
                }

                // Now the ICMP part
                icp = (struct icmp *)(packet + hlen);
                if (icp->icmp_type == ICMP_ECHOREPLY)
                {
                    //cout << "Recv: echo reply"<< endl;
                    if (icp->icmp_seq != k)
                    {
                        // cout << "received sequence # " << icp->icmp_seq << endl;
                        continue;
                    }
                    if (icp->icmp_id != my_tid % 65000)
                    {
                        //   cout << "received id " << icp->icmp_id << endl;
                        continue;
                    }
                    cont = false;
                }
                else
                {
                    // cout << "Recv: not an echo reply" << endl;
                    continue;
                }

                end_t = (double)(1000000000*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec));

                // auto end_T = steady_clock::now();
                // double elapsed_T = ((end_T-start_T).count())*steady_clock::period::num / static_cast<double>(steady_clock::period::den);
                sumRTTs +=end_t;
                break;
            }
            else
            {

                clock_gettime(CLOCK_MONOTONIC, &end); //use CLOCK_MONOTONIC in deployment
                end_t = (double)(1000000000*(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec));
                //cout <<end_t<< "   ping timed-out. "+targetIP+"\n";
                sumRTTs +=end_t;
                break;
            }
        }
    }
    close(s);
    free(packet);
    return (sumRTTs/double(1024))/1000000000.0; //sec
}



uint16_t parPinger::in_cksum(uint16_t *addr, unsigned len)
{
    uint16_t answer = 0;
    /*
       * Our algorithm is simple, using a 32 bit accumulator (sum), we add
       * sequential 16 bit words to it, and at the end, fold back all the
       * carry bits from the top 16 bits into the lower 16 bits.
       */
    uint32_t sum = 0;
    while (len > 1)  {
        sum += *addr++;
        len -= 2;
    }

    // mop up an odd byte, if necessary
    if (len == 1) {
        *(unsigned char *)&answer = *(unsigned char *)addr ;
        sum += answer;
    }

    // add back carry outs from top 16 bits to low 16 bits
    sum = (sum >> 16) + (sum & 0xffff); // add high 16 to low 16
    sum += (sum >> 16); // add carry
    answer = ~sum; // truncate to 16 bits
    return answer;
}


//Sends a burst of ICMP Echo requests to the targetIP at a rate of 0.5*meanRTT
int parPinger::send_probe(string target_ip, vector<bool> mls_seq, uint16_t scanner_id)
{
    /* Open Sending Socket */
    int s, i, cc, packlen, datalen = 1500;
    struct hostent *hp;
    struct sockaddr_in to, from;
    //struct protoent	*proto;
    struct ip *ip;
    u_char *packet, outpack[MAXPACKET];
    string hostname;
    struct icmp *icp;


    to.sin_family = AF_INET;

    // try to convert as dotted decimal address, else if that fails assume it's a hostname
    to.sin_addr.s_addr = inet_addr(target_ip.c_str());
    if (to.sin_addr.s_addr != (u_int)-1)
        hostname = target_ip.c_str();
    else
    {
        cerr << "unknown host "<< target_ip << endl;
        return -1;
    }
    packlen = datalen + MAXIPLEN + MAXICMPLEN;
    if ( (packet = (u_char *)malloc((u_int)packlen)) == NULL)
    {
        cerr << "malloc error\n";
        return -1;
    }

    if ( (s = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0)
    {
        perror("socket");	/* probably not running as superuser */
        free(packet);
        return -1;
    }

    /* Execute Burst */
    icp = (struct icmp *)outpack;
    icp->icmp_type = ICMP_ECHO;
    icp->icmp_code = 0;
    icp->icmp_cksum = 0;
    icp->icmp_seq = 0;	/* seq and id must be reflected */
    icp->icmp_id = scanner_id;
    cc = datalen + ICMP_MINLEN;
    icp->icmp_cksum = in_cksum((unsigned short *)icp,cc);
    struct timespec tmp;

    struct timespec start;
    struct timespec stop;
    struct timespec i_start;i_start.tv_sec=0;i_start.tv_nsec=0;
    struct timespec i_stop;i_stop.tv_sec=0;i_stop.tv_nsec=0;
    struct timespec wait_time;wait_time.tv_sec=0;wait_time.tv_nsec=0;


    clock_gettime(CLOCK_MONOTONIC, &start); //start pir time
    int count = 0;
    point tx;

    for(int k = 0; k < mls_seq.size(); k++)
    {
        //calc wait time
        wait_time.tv_nsec = max(ping_interval.tv_nsec - (i_stop.tv_nsec - i_start.tv_nsec) - 70000, 0L );
        timespec_diff(i_start,i_stop,ping_interval,wait_time);

        //wait_time.tv_nsec = 0L ;//trying to account for processing time and timer accuracy

        //wait
        nanosleep(&wait_time,&tmp);

        //setup ping
        cc = ((int)mls_seq[k])*1500 + ICMP_MINLEN;

        //update checksum
        icp->icmp_cksum = 0;
        icp->icmp_cksum = in_cksum((unsigned short *)icp,cc);
        //send
        clock_gettime(CLOCK_MONOTONIC, &i_start);
        i = sendto(s, (char *)outpack, cc, 0, (struct sockaddr*)&to, (socklen_t)sizeof(struct sockaddr_in));
        tx.indx = icp->icmp_seq;
        tx.t = i_start;
        send_times.push_back(tx);

        if (i < 0 || i != cc)
        {
            if (i < 0)
                perror("sendto error");
            cout << "wrote " << hostname << " " <<  cc << " chars, ret= " << i << endl;
        }
        //increment seq
        icp->icmp_seq++;

        count++;
//        if(count%100==0){
//            cout<<"*";
//            cout.flush();
//        }
        clock_gettime(CLOCK_MONOTONIC, &i_stop);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop); //reception time
    //cout<<endl<<endl;

    /* Close Sending Socket */
    close(s);
    free(packet);
    double start1 = (double)(start.tv_sec) + (double)(start.tv_nsec)/1000000000;
    double stop1 = (double)(stop.tv_sec) + (double)(stop.tv_nsec)/1000000000;

    burstTime = stop1-start1;

    return 0;
}

//function for computing send interval based on overhead
void parPinger::timespec_diff(struct timespec &startOverhead, struct timespec &stopOverhead, struct timespec &sendInterval, struct timespec &computed_sleepInterval)
{
    computed_sleepInterval.tv_sec = sendInterval.tv_sec;
    computed_sleepInterval.tv_nsec = sendInterval.tv_nsec;

    //Compute overhead
    struct timespec overhead;
    if ((stopOverhead.tv_nsec - startOverhead.tv_nsec) < 0) {
        overhead.tv_sec = stopOverhead.tv_sec - startOverhead.tv_sec - 1;
        overhead.tv_nsec = stopOverhead.tv_nsec - startOverhead.tv_nsec + 1000000000;
    } else {
        overhead.tv_sec = stopOverhead.tv_sec - startOverhead.tv_sec;
        overhead.tv_nsec = stopOverhead.tv_nsec - startOverhead.tv_nsec;
    }


    //compute interval
    if ((stopOverhead.tv_nsec - startOverhead.tv_nsec) < 0) {
        computed_sleepInterval.tv_sec = computed_sleepInterval.tv_sec - overhead.tv_sec - 1;
        computed_sleepInterval.tv_nsec = computed_sleepInterval.tv_nsec - overhead.tv_nsec + 1000000000;
    } else {
        computed_sleepInterval.tv_sec = computed_sleepInterval.tv_sec - overhead.tv_sec;
        computed_sleepInterval.tv_nsec = computed_sleepInterval.tv_nsec - overhead.tv_nsec;
    }

    if(computed_sleepInterval.tv_sec < 0 || computed_sleepInterval.tv_nsec < 0)
    {
        computed_sleepInterval.tv_sec = 0;
        computed_sleepInterval.tv_nsec = 0;
    }
    return;
}



}