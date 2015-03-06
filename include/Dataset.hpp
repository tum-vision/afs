#ifndef DATASET_HPP
#define DATASET_HPP

#include <common.hpp>
#include <Feature.hpp>
#include <Learning.hpp>
#include <Potential.hpp>
#include <Optimization.hpp>

/**
 * @brief The Dataset class
 */
class Dataset{

public:
    /**
     * @brief Class Constructor
     */
    Dataset();
    /**
     * @brief Class Deconstructor
     */
    virtual ~Dataset();

    /**
     * @brief Convert from RGB to Label
     * @param rgb RGB image
     * @param labels label matrix to store labels for each pixel
     */
    virtual void RGB2Label(Mat rgb, Mat& labels);

    /**
     * @brief Convert from label to RGB
     * @param labels label matrix
     * @param rgb    RGB image
     */
    virtual void Label2RGB(Mat labels, Mat& rgb);

    /**
     * @brief Split Dataset
     * @param trainCount number of training samples, either a number or a proportion
     * @param testCount number of test samples, either a number or a proportion
     */
    virtual void getDatasetImages(int trainCount, int testCount);

    /**
     * @brief Read Dataset
     * @param imgs  vector of image names
     * @param folder folder to read images from
     * @param ext extension of image files
     */
    virtual void readImgsInDir(vector<string>& imgs, string folder, string ext);

    /**
     * @brief Read pre-splitted train/validation/test images
     * @return true if split files exist
     */
    virtual bool readSplitsToVectors();

    /**
     * @brief Write train/validation/test splits to files
     */
    virtual void writeSplitsToFile();

    /**
     * @brief Create directory
     * @param dirName name of directory
     * @return 1 if directory exist
     */
    virtual int  createDir(string dirName);

    /**
     * @brief Remove all file in directory
     * @param dirName directory to be cleared
     * @return 1 if all files succesfully removed
     */
    virtual int  clearDir(string dirName);

    /**
     * @brief Find image index in testImgs
     * @param name name of the image
     * @return index of the image in testImgs
     */
    virtual int  findImage(string name);

    /**
     * @brief Create Folders for Results
     */
    virtual void createResultFolders();

    /**
     * @brief Get size of full feature set
     * @return size of full feature set
     */
    virtual int  getFullFeatureDim();

    /**
     * @brief Remove Unlabeled Images from Dataset
     */
    virtual void removeUnlabeledImages();

protected:
    /**
     * @brief variable to store index
     */
    int index;

public:

    /**
     * @brief list of all images in dataset
     */
    vector<string> allImages;

    /**
     * @brief list of training images
     */
    vector<string> trainImgs;

    /**
     * @brief list of test images
     */
    vector<string> testImgs;

    /**
     * @brief list of validation images
     */
    vector<string> validationImgs;

    /**
     * @brief vector stores indices of selected features
     */
    vector<int>     selectedFeatures;

    /**
     * @brief flag to split dataset randomly
     */
    bool    isSplitDataset;

    /**
     * @brief flag to save potentials
     */
    bool    savePotentials;

    /**
     * @brief flag to save detection results (argmax of potential for each pixel on image)
     */
    bool    saveDetections;

    /**
     * @brief flag to visualize entropy map of the image
     */
    bool    showEntropy;

    /**
     * @brief size of the training set
     */
    int     trainCount;

    /**
     * @brief size of the test test
     */
    int     testCount;

    /**
     * @brief entopy of the training data (balance of the classes in training set)
     */
    double  trainEntropy;

    /**
     * @brief entropy of the test set (how certain the detection is)
     */
    double  testEntropy;

    /**
     * @brief name of the dataset
     */
    string  name_;

    /**
     * @brief folder to store the segmentation outputs
     */
    string  resultDir;

    /**
     * @brief folder to store classifiers
     */
    string  trainFolder;

    /**
     * @brief folder to store results. (e.g. mainFolder + '/Result')
     */
    string  resultFolder;

    /**
     * @brief main folder of dataset
     */
    string  mainFolder;

    /**
     * @brief folder where RGB image files are
     */
    string  imageFolder;

    /**
     * @brief extension of RGB image files
     */
    string  imageFileExt;

    /**
     * @brief folder where ground truth files are
     */
    string  grFolder;

    /**
     * @brief extension of ground truth files
     */
    string  grFileExt;

    /**
     * @brief folder to store depth features
     */
    string  depthFolder;

    /**
     * @brief extension of depth feature files
     */
    string  depthFileExt;

    /**
     * @brief folder to store haar-like features
     */
    string  haarFolder;

    /**
     * @brief extension of haar-like feature files
     */
    string  haarFeatureExt;

    /**
     * @brief folder to store color features
     */
    string  colorFolder;

    /**
     * @brief extension of color feature files
     */
    string  colorFeatureExt;

    /**
     * @brief folder to store location features
     */
    string  locationFolder;

    /**
     * @brief extension of location feature files
     */
    string  locationFeatureExt;

    /**
     * @brief folder to store texton features
     */
    string  textonFolder;

    /**
     * @brief extension of texton feature files
     */
    string  textonFeatureExt;

    /**
     * @brief folder to store depth features
     */
    string  depthFeatureFolder;

    /**
     * @brief extension of depth feature files
     */
    string  depthFeatureExt;

    /**
     * @brief folder to stor potentials
     */
    string  potentialFolder;

    /**
     * @brief extension of potential files
     */
    string  potentialExt;

    /**
     * @brief folder to store detection results
     */
    string  detectionFolder;

    /**
     * @brief folder to store potential maps
     */
    string  pMapFolder;


    /**
     * @brief total size of current feature set
     */
    int    numFeatureDims;

    /**
     * @brief Number of features in the current(selected) feature set
     */
    int    fcount_;


    /**
     * @brief maximum number of regions (== numClasses)
     */
    int n;

    /**
     * @brief maximum iteration number for VariationalOptimization
     */
    int maxSteps;

    /**
     * @brief smoothing parameter lambda for VariationalOptimization
     */
    float lambda;

    /**
     * @brief sub sample size
     */
    int subSample;

    /**
     * @brief bandwidth of kernels for TextonFeautures
     */
    double  textonBandWidth;

    /**
     * @brief total number of classes in Dataset
     */
    unsigned int numClasses;

    /**
     * @brief instance of HaarLikeFeatures
     */
    HaarLikeFeatures    *haarFeatures;

    /**
     * @brief instance of ColorFeatures
     */
    ColorFeatures       *colorFeatures;

    /**
     * @brief instance of LocationFeatures
     */
    LocationFeatures    *locationFeatures;

    /**
     * @brief instance of TextonFeatures
     */
    TextonFeatures      *textonFeatures;

    /**
     * @brief instance of DepthFeatures
     */
    DepthFeatures       *depthFeatures;

    /**
     * @brief Weights of each class to balance the training error
     */
    const float *class_weights;

    /**
     * @brief Structure to store parameteres for Random Forest
     */
    CvRTParams      RFParams;

    /**
     * @brief termination criteria for Random Forests training
     */
    CvTermCriteria  RFterm_crit;

    /**
     * @brief maximum number of trees in the forest
     */
    int   RFmax_num_of_trees_in_the_forest;
    /**
     * @brief sufficient accuracy, OOB error (OpenCV docs)
     */
    float RFforest_accuracy;

    /**
     * @brief total number of feature types
     */
    int totalFeatureType;

    /**
     * @brief number of potentails
     */
    int numPotentials;

    /**
     * @brief pointer to the list of features
     */
    Feature **features;

    /**
     * @brief pointer to the list of potentials
     */
    Potential **potentials;

    /**
     * @brief pointer to the optimizer
     */
    Optimization *optimizer;
};

/**
 * @brief The eTrims class
 */
class eTrims: public Dataset{

    // window     1, vegetation      2, sky         3, building       4
    // car        5, road            6, door        7, pavement       8

public:
    /**
     * @brief eTrims
     */
    eTrims();
    void RGB2Label(Mat rgb, Mat &labels);
    void Label2RGB(Mat labels, Mat&rgb);

};

/**
 * @brief The MSRC class
 */
class MSRC: public Dataset{

    // building  0, grass     1, tree     2, cow       3
    // horse     4, sheep     5, sky      6, mountain  7
    // plane     8, water     9, face    10, car      11
    // bike     12, flower   13, sign    14, bird     15
    // book     16, chair    17, road    18, cat      19
    // dog      20, body     21, boat    22

public:
    /**
     * @brief MSRC
     */
    MSRC();
    void RGB2Label(Mat rgb, Mat& labels);
    void Label2RGB(Mat labels, Mat& rgb);
};

/**
 * @brief The Corel class
 */
class Corel: public Dataset{

    // rhino/hippo 0, polarbear 1, water 2, snow    3
    // vegetation  4, ground    5, sky   6

public:
    /**
     * @brief Class Constructor
     */
    Corel();
    void RGB2Label(Mat rgb, Mat& labels);
    void Label2RGB(Mat labels, Mat& rgb);
};

/**
 * @brief The Sowerby class
 */
class Sowerby : public Dataset{

    // sky      0, grass   1, roadline 2, road    3
    // building 4, sign    5, car      6

public :
    /**
     * @brief Class Constructor
     */
    Sowerby();

};

/**
 * @brief The NYUv1 class
 */
class NYUv1: public Dataset{

    // bed      1, blind  2,  bookshelf 3,   cabinet    4
    // ceiling  5, floor  6,  picture   7,   sofa       8
    // table    9, tv     10, wall      11,  window    12

public:
    /**
     * @brief Class Constructor
     */
    NYUv1();
    void RGB2Label(Mat rgb, Mat &labels);
    void Label2RGB(Mat labels, Mat&rgb);

};

/**
 * @brief The NYUv2 class
 */
class NYUv2: public Dataset{

    // bed      1, objects  2, chair    3,   furniture 4
    // ceiling  5, floor    6, deco     7,   sofa      8
    // table    9, wall    10, window  11,   books    12
    // tv      13

public:
    /**
     * @brief Class Constructor
     */
    NYUv2();
    void RGB2Label(Mat rgb, Mat &labels);
    void Label2RGB(Mat labels, Mat&rgb);

};

/**
 * @brief The VOC2012 class
 */
class VOC2012: public Dataset{

    // aeroplane     1, bicycle  2, bird   3, boat        4, bottle      5
    // bus           6, car      7, cat    8, chair       9, cow        10
    // diningtable  11, dog     12, horse 12, motorbike  13, person     15
    // potted plant 16, sheep   17, sofa  18, train      19, tv/monitor 20

    // background    0, void/unlabeled 255

public:
    /**
     * @brief Class Constructor
     */
    VOC2012();
    void RGB2Label(Mat rgb, Mat& labels);
    void Label2RGB(Mat labels, Mat& rgb);
};

/**
 * @brief The PedestrianParsing class
 */
class PedestrianParsing : public Dataset{

    // background   1, hair      2, face   3, pullover     4
    // arms         5, trousers  6, legs   7, shoes        8

public :
    /**
     * @brief Class Constructor
     */
    PedestrianParsing();
    void RGB2Label(Mat rgb, Mat& labels);
    void Label2RGB(Mat labels, Mat& rgb);

};

/**
 * @brief The Test class
 */
class Test: public Dataset{

    // sky      0, grass   1, roadline 2, road    3
    // building 4, sign    5, car      6

public :
    /**
     * @brief Class Constructor
     */
    Test();
};

#endif // DATASET_HPP
