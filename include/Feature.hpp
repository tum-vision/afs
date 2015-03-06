#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <Image.hpp>
#include <common.hpp>

/**
 * @brief The Feature class
 */
class Feature {

public:
    /**
     * @brief Class Constructor
     */
    Feature();
    /**
     * @brief Class Deconstructor
     */
    virtual ~Feature(){};

    /**
     * @brief Save features
     * @param imName image name
     * @param features feature matrix
     */
    void         saveFeature(string imName, Mat& features);

    /**
     * @brief Load feature
     * @param imName image name
     * @param features feature matrix
     */
    void         loadFeature(string imName, Mat& features);

    /**
     * @brief Extract features
     * @param im    image
     * @param features feature matrix
     */
    virtual void extractFeatures(Image* im, Mat& features)=0;

    /**
     * @brief Set sub sample size
     * @param ss sub sample size
     */
    void         setSubSample(int ss);

    /**
     * @brief Get sub sample size
     * @return size of sub sample
     */
    int          getSubSample();

    /**
     * @brief Get size of feature set
     * @return size of feature set
     */
    int          getFeatureDim();  // To get current size of feature vector

    /**
     * @brief Get size of full feature set
     * @return size of full feature set
     */
    int          getFullDim();     // To get full size of the initial feature vector

    /**
     * @brief Set size of feature set
     * @param dim size
     */
    void         setFeatureDim(int dim);

protected:

    /**
     * @brief File operator
     */
    FILE *foperator;

    /**
     * @brief folder to store feature
     */
    string  ftFolder;

    /**
     * @brief feature file extension
     */
    string  ext_;

    /**
     * @brief size of sub sample
     */
    int subSample;

    /**
     * @brief size of feature
     */
    int featureSize;

    /**
     * @brief size of selected feature set
     */
    int featureDim;

    /**
     * @brief patch height
     */
    int rHeight;

    /**
     * @brief patch width
     */
    int rWidth;

    /**
     * @brief half size of patch height
     */
    int halfHeight;

    /**
     * @brief half size of patch width
     */
    int halfWidth;

    /**
     * @brief sixth of patch height
     */
    int sixthOfHeight;

    /**
     * @brief sixth of patch width
     */
    int sixthOfWidth;

    /**
     * @brief fourth of patch height
     */
    int fourthOfHeight;

    /**
     * @brief fourth of patch width
     */
    int fourthOfWidth;

    /**
     * @brief weight of black region (Haar-like features)
     */
    float weight0;

    /**
     * @brief weight of white region (Haar-like features)
     */
    float weight1;

    /**
     * @brief vector of all patches
     */
    vector<Size> allPatches;

    /**
     * @brief timer to estimate the time
     */
    Timer timer;

    /**
     * @brief Set rectangle size
     * @param height height of new rectangle
     * @param width  width of new rectangle
     */
    void setRectSize(int height, int width);

public:

    /**
     * @brief name of the feature
     */
    string name_;

    /**
     * @brief matrix to store feature indices indicate which features to be computed
     */
    Mat isComputeFeature;

};

/**
 * @brief The HaarLikeFeatures class
 */
class HaarLikeFeatures : public Feature{


public:

    /**
     * @brief Class Constructor
     * @param all_patches set of all patches
     * @param haarFolder folder to store features
     * @param ext extension of feature files
     * @param numSubSample sub sample size
     */
    HaarLikeFeatures(vector<Size>& all_patches, string haarFolder, string ext,int  numSubSample);
    void extractFeatures(Image* im, Mat& features);

protected:

    /**
     * @brief Horizontal Edge Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void horizontalEdge(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Vertical Edge Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void verticalEdge(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Horizontal Line Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void horizontalLine(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Vertical Edge Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void verticalLine(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Center Surround Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void centerSurround(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Four Square Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void fourSquare(Mat&im, int r, int c,float& fValue);

};

/**
 * @brief The ColorFeatures class
 */
class ColorFeatures : public Feature{

public:
    /**
     * @brief relative pixel positions
     */
    vector<Point2d> rndLocs;

    /**
     * @brief relative pixel positions
     */
    vector<Point2d> rndRelatives;

    /**
     * @brief relative patch positions
     */
    vector<Rect>    rndPatches;     // for Relative Patches
public:
    /**
     * @brief Class Constructor
     * @param all_patches set of all patches
     * @param colorFolder folder to store features
     * @param ext extension of feature files
     * @param numSubSample sub sample size
     */
    ColorFeatures(vector<Size>& all_patches, string colorFolder, string ext,int numSubSample);
    void extractFeatures(Image* im, Mat& features);

private:
    /**
     * @brief Pixel Color Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void pixelColor(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Relative Color Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param i index of relative color position in rndLocs
     * @param fValue feature value
     */
    void relativeColor(Mat&im, int r, int c, int i, float& fValue);

    /**
     * @brief Realtive Color Comparison Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param i index of relative color position in rndRelatives
     * @param fValue feature value
     */
    void relativeColorComp(Mat&im, int r, int c, int i, float& fValue);

    /**
     * @brief Relative Patch Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param i index of relative color position in rndPatches
     * @param fValue feature value
     */
    void relativePatch(Mat&im, int r, int c, int i, float& fValue);

    /**
     * @brief Horizontal Color Edge Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void horizontalColorEdge(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Vertical Color Edge Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void verticalColorEdge(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Center Surround Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void centerSurround(Mat&im, int r, int c, float &fValue);

};


/**
 * @brief The LocationFeatures class
 */
class LocationFeatures : public Feature{

public:
    /**
     * @brief Class Constructor
     * @param LocFolder folder to store features
     * @param ext extension of feature files
     * @param numSubSample sub sample size
     */
    LocationFeatures(string LocFolder, string ext,int numSubSample);
    void extractFeatures(Image* im, Mat& features);

};

/**
 * @brief The DepthFeatures class
 */
class DepthFeatures : public Feature{

public:
    /**
     * @brief Class Constructor
     * @param all_patches set of all patches
     * @param depthFolder folder to store features
     * @param ext extension of feature files
     * @param numSubSample sub sample size
     */
    DepthFeatures(vector<Size>& all_patches, string depthFolder, string ext,int numSubSample);
    void extractFeatures(Image* im, Mat& features);

private:
    /**
     * @brief Matrix to store maximum depth of each column
     */
    Mat                 maxIncolumn;

    /**
     * @brief relative pixel positions
     */
    vector<Point2d>     rndRelatives;
private:
    /**
     * @brief Relative Depth Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void relativeDepth(Mat&im, int r, int c,float& fValue);

    /**
     * @brief Point Height Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param fValue feature value
     */
    void pointHeight(Mat &im, int r, int c, float &fValue);

    /**
     * @brief Depth Comparison Feature
     * @param im image
     * @param r row of pixel at which the feature will be computed
     * @param c column of pixel at which the feature will be computed
     * @param i index of relative color position in rndRelatives
     * @param fValue feature value
     */
    void depthComp(Mat &im, int r, int c, int i, float &fValue);
};

/**
 * @brief The TextonFeatures class
 */
class TextonFeatures : public Feature{

public:
    /**
     * @brief Class Constructor
     * @param TextonFolder folder to store features
     * @param ext extension of feature files
     * @param numSubSample sub sample size
     * @param bandWidth bandwidth of texton kernels
     */
    TextonFeatures(string TextonFolder, string ext, int numSubSample, double bandWidth);
    void extractFeatures(Image* im, Mat& filterResponses);

private:
    /**
     * @brief relative patch positions
     */
    vector<Rect>    rndPatches;     // for Relative Patches

    /**
     * @brief Kernel Bandwidth
     */
    double kappa;
};

#endif // FEATURE_HPP
