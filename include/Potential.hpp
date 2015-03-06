#ifndef POTENTIAL_HPP
#define POTENTIAL_HPP

#include <common.hpp>
#include <Learning.hpp>

class Dataset;

/**
 * @brief The Potential class
 */
class Potential
{
public:

    /**
     * @brief Class Constructor
     */
    Potential(){};

    /**
     * @brief Class Deconstructor
     */
    virtual ~Potential(){};

    /**
     * @brief Train potential
     * @param dataset   pointer to the current dataset
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to process all images startin from "from"
     * @param FAST_COMPUTATION  flag to activate fast computation
     * @return  estimated training time in miliseconds
     */
    virtual int Train(Dataset* dataset,vector<string>& imageList,int from, int to, bool FAST_COMPUTATION)=0;

    /**
     * @brief Evaluate potentials
     * @param dataset           pointer to the current dataset
     * @param imageList         list of images
     * @param from              start index
     * @param to                stop index, set to -1 to process all images startin from "from"
     * @param FAST_COMPUTATION  flag to activate fast computation
     * @return  estimated time per image in miliseconds
     */
    virtual int Evaluate(Dataset* dataset,vector<string>& imageList,int from, int to, bool FAST_COMPUTATION)=0;

    /**
     * @brief Save potential
     * @param imName        path to the image
     * @param potentials    potentials
     * @param classNo       total number of classes
     */
    void    savePotential(string imName, Mat& potentials, int classNo);

    /**
     * @brief Load Potential
     * @param imName        path to the image
     * @param potentials    potentials
     */
    void    loadPotential(string imName, Mat& potentials);

public:

    /**
     * @brief Potential name
     */
    string name_;

    /**
     * @brief pointer to the learning algorithm
     */
    Learning* learning;

protected:

    /**
     * @brief file operator
     */
    FILE *foperator;

    /**
     * @brief potential output folder
     */
    string folder;

    /**
     * @brief potential extension
     */
    string ext;

    /**
     * @brief timer to estimate the time
     */
    Timer   timer;
};

/**
 * @brief The DenseUnaryPixelPotential class
 */
class DenseUnaryPixelPotential: public Potential{

public:
    /**
     * @brief Class Constructior
     * @param dir   Output directory
     * @param fileExt   file extension
     */
    DenseUnaryPixelPotential(string dir, string fileExt);

    int Train(Dataset* dataset,vector<string>& imageList,int from, int to, bool FAST_COMPUTATION = false);
    int Evaluate(Dataset* dataset,vector<string>& imageList,int from, int to, bool FAST_COMPUTATION = false);
};

#endif // POTENTIAL_HPP
