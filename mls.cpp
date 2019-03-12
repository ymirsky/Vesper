#include "mls.h"

// n_bits detemines the length of each MLS sequence (n_bits=10 produces sequences of 1023 bits)
// useAES=true tells the class that AES256-CTR should be used as the seed for each sequence. If useAES-false, then std::random_device is used.
//AES should be used to produce a CSPRNG if you have doubt wheter std::random_device is producing non-deterministicly random values.
mls::mls(int n_bits, bool useAES)
{
    setBits(n_bits);
    isAES = useAES;
    if(isAES){
        key = new uint8_t[32];
        iv = new uint8_t[16];
        for(int i = 0; i < 32; i++)
            key[i] = rd();
        for(int i = 0; i < 16; i++)
            iv[i] = rd();
        AES_init_ctx_iv(&ctx, key, iv);
       }
}


void mls::setBits(int n_bits)
{
    if(n_bits < 2)
        nbits = 2;
    else if(n_bits > 32)
        nbits = 32;
    else
        nbits = n_bits;

    //define tap points
    Taps[2] = {1};
    Taps[3] = {2};
    Taps[4] = {3};
    Taps[5] = {3};
    Taps[6] = {5};
    Taps[7] = {6};
    Taps[8] = {7, 6, 1};
    Taps[9] = {5};
    Taps[10] = {7};
    Taps[11] = {9};
    Taps[12] = {11, 10, 4};
    Taps[13] = {12, 11, 8};
    Taps[14] = {13, 12, 2};
    Taps[15] = {14};
    Taps[16] = {15, 13, 4};
    Taps[17] = {14};
    Taps[18] = {11};
    Taps[19] = {18, 17, 14};
    Taps[20] = {17};
    Taps[21] = {19};
    Taps[22] = {21};
    Taps[23] = {18};
    Taps[24] = {23, 22, 17};
    Taps[25] = {22};
    Taps[26] = {25, 24, 20};
    Taps[27] = {26, 25, 22};
    Taps[28] = {25};
    Taps[29] = {27};
    Taps[30] = {29, 28, 7};
    Taps[31] = {28};
    Taps[32] = {31, 30, 10};
}

int mls::size()
{
    return pow(2,nbits)-1;
}

vector<bool> mls::get_seq()
{
    vector<int> taps = Taps[nbits];
    vector<bool> state(nbits,false);
    int length=pow(2,nbits) - 1;
    vector<bool> seq(length);
    bool feedback;

    do{
        uint32_t randomNum;
        if(isAES)
        {
            //prepare message
            uint8_t in[64];
            for(int i = 4; i < 64; i++)
            {
                in[i] = rd();
            }
            //encrypt message
            AES_CTR_xcrypt_buffer(&ctx, in, 64);
            randomNum = (uint32_t)((in[3]<<24)|(in[2]<<16)|(in[1]<<8)|(in[0])); //first 32 bits of the cipher stream
        }
        else
        {
            randomNum = rd();//use device entropy to genrate random number
        }
        for (int i = 0;i< nbits;i++)
            state[i] = (randomNum >> i) & 1;


        int idx = 0;

        for( int i = 0; i< length; i++)
        {
            feedback = state[idx];
            seq[i] = feedback;
            for(int ti = 0; ti < taps.size();ti++)
                feedback ^= state[(taps[ti] + idx) % nbits];
            state[idx] = feedback;
            idx = (idx + 1) % nbits;
        }
    }while(!goodSeq(seq));//try again in the rare case that a sequnce of zeros are generated
    return seq;
}

bool mls::goodSeq(vector<bool> seq){
    int numZeros = 0;
    for(int i =0; i < seq.size(); i++)
        if(seq[i]==false)
            numZeros++;
    if(numZeros==seq.size())
        return false;
    return true;
}
