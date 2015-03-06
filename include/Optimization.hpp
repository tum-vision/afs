#ifndef OPTIMIZATION_H
#define OPTIMIZATION_H

#include <common.hpp>

class Dataset;

/**
 * @brief The Optimization class
 */
class Optimization
{
public:
    /**
     * @brief optimize solution
     * @param dataset   pointer to the current dataset
     * @param imageList list of images to segment
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     * @param printTime flag to print out the estiamted time on output
     * @return          estimated time per image in miliseconds
     */
    virtual int Solve(Dataset* dataset,vector<string>& imageList,int from, int to, bool printTime = true)=0;

};

/**
 * @brief The VariationalOptimization class
 */
class VariationalOptimization : public Optimization{

private:
    /**
     * @brief totalTime total time to segment all images
     */
    int   totalTime;

    /**
     * @brief timer to estimate the time
     */
    Timer timer;
public:
    /**
     * @brief Class Constructor
     */
    VariationalOptimization();

    int Solve(Dataset* dataset,vector<string>& imageList,int from, int to, bool printTime = true);

};

#endif // OPTIMIZATION_H
