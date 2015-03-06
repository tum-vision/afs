#ifndef COMMON_HPP
#define COMMON_HPP

// System Includes
#include <exception>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <typeinfo>


// Optimization
#include <istream>
#include <sstream>
#include <cuda/segmentation.cuh>
#include <cuda/binarization.cuh>
#include <fcntl.h>

// OPENCV INCLUDES
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/gpu/gpu.hpp>

// BOOST INCLUDES
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;
using namespace std;
using namespace cv;

#define CLOCKS_PER_MS (CLOCKS_PER_SEC / 1000)

/**
 * @brief The Timer class
 */
class Timer{

private:
    /**
     * @brief clock start
     */
    clock_t     init;

    /**
     * @brief time value
     */
    struct timeval tv;

    /**
     * @brief start time value
     */
    struct timeval start_tv;

    /**
     * @brief hours
     */
    int hours;

    /**
     * @brief minutes
     */
    int mins;

    /**
     * @brief seconds
     */
    int secs;

    /**
     * @brief miliseconds
     */
    int msecs;

    /**
     * @brief remaining miliseconds after time conversion
     */
    int last_msecs;

public:

    /**
     * @brief start clock
     */
    void start(){

        init = clock();
        gettimeofday(&start_tv, NULL);
    }

    /**
     * @brief stop clock
     * @param msg  message to print out
     * @return time in miliseconds
     */
    int stop(string msg= ""){

        gettimeofday(&tv, NULL);
        msecs = (tv.tv_sec - start_tv.tv_sec)   * 1000.0 +
                (tv.tv_usec - start_tv.tv_usec) / 1.0e6;

        secs    = (int) (msecs   / 1000        ) % 60 ;
        mins    = (int) ((msecs  / (1000*60   )) % 60);
        hours   = (int) ((msecs  / (1000*60*60)) % 24);

        last_msecs   = msecs - hours * (1000*60*60) - mins * (1000*60) - secs * 1000;

        init   = 0;
        tv.tv_sec    = 0;
        tv.tv_usec   = 0;

        if(!msg.empty())
            cout<<msg<<hours<<" hours, "<<mins<<" minutes, "<<secs<<" seconds, "<<last_msecs<<" milliseconds."<<endl;

        return msecs;
    }

    /**
     * @brief print out message
     * @param msecs miliseconds
     * @param msg   message
     */
    void msg(int msecs, string msg){

        int secs    = (int) (msecs  / 1000        ) % 60;
        int mins    = (int) ((msecs / (1000*60   )) % 60);
        int hours   = (int) ((msecs / (1000*60*60)) % 24);

        int last_msecs = msecs - hours * (1000*60*60) - mins * (1000*60) - secs * 1000;

        cout<<msg<<hours<<" hours, "<<mins<<" minutes, "<<secs<<" seconds, "<<last_msecs<<" milliseconds."<<endl;
    }

    /**
     * @brief format time
     * @param msecs miliseconds
     * @return  formatted time
     */
    static string formatted_time(int msecs){

        int secs    = (int) (msecs  / 1000        ) % 60;
        int mins    = (int) ((msecs / (1000*60   )) % 60);
        int hours   = (int) ((msecs / (1000*60*60)) % 24);

        int last_msecs = msecs - hours * (1000*60*60) - mins * (1000*60) - secs * 1000;

        return (std::string(2-num_digits(hours), '0') + to_string(hours)+
                std::string(2-num_digits(mins), '0')  + to_string(mins)+
                std::string(2-num_digits(secs), '0')  + to_string(secs)+
                std::string(3-num_digits(last_msecs), '0') + to_string(last_msecs));
    }

    /**
     * @brief parse formatted time
     * @param time_stream   time stream
     * @return  time in miliseconds
     */
    static int parse_formatted_time(string time_stream){

        return ((atoi(time_stream.substr(0,2).c_str())* 1000 * 60 * 60  ) +
                (atoi(time_stream.substr(2,2).c_str())* 1000 * 60       ) +
                (atoi(time_stream.substr(4,2).c_str())* 1000            ) +
                (atoi(time_stream.substr(6,3).c_str()))
                );
    }

    /**
     * @brief number of digits
     * @param num   number
     * @return total number of digits
     */
    static int num_digits(int num){

            int digits = 0;
            if (num == 0) digits = 1; // remove this line if '-' counts as a digit
            while (num) {
                num /= 10;
                digits++;
            }
            return digits;
    }

    /**
     * @brief convert number to string
     * @param num   number
     * @return  string
     */
    static string to_string(int num){

        stringstream ss;
        ss << num;
        return ss.str();
    }
};

/**
 * @brief The Performance class
 */
class Performance{

public:

    /**
     * @brief Class Constructor
     * @param init  initial performance score
     */
    Performance(double init=0){
        overall = average = waverage = init;
    }

    /**
     * @brief set performance
     * @param perf  performance
     */
    void set(Performance perf){
        this->overall   = perf.overall;
        this->average   = perf.average;
        this->waverage  = perf.waverage;
    }

    /**
     * @brief operator =
     * @param p     pointer to performance
     * @return      new performance
     */
    Performance& operator=(const Performance& p)
    {
        this->overall       = p.overall;
        this->average       = p.average;
        this->waverage      = p.waverage;
        return *this;
    }

    /**
     * @brief operator +=
     * @param p     pointer to performance
     * @return      new performance
     */
    Performance& operator+=(const Performance& p)
    {
        this->overall     += p.overall;
        this->average     += p.average;
        this->waverage    += p.waverage;
        return *this;
    }

    /**
     * @brief operator /=
     * @param t     number
     * @return      new performance
     */
    Performance& operator/=(const int& t)
    {
        this->overall     /= t;
        this->average     /= t;
        this->waverage    /= t;
        return *this;
    }
public:
    /**
     * @brief overall score
     */
    double overall;

    /**
     * @brief average score
     */
    double average;

    /**
     * @brief weighted average score
     */
    double waverage;
};

/**
 * @brief list of feature_names
 */
extern vector<string>feature_names_;

/**
 * @brief edge detection function used in VariationalOptimization
 * @param I_Vec3b   input image
 * @param g         edge image
 */
void edgeDetectionFct(cv::Mat I_Vec3b, cv::Mat &g);

#endif // COMMON_HPP
