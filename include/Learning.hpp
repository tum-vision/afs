#ifndef LEARNING_HPP
#define LEARNING_HPP

#include <common.hpp>
#include <exception>

/**
 * @brief Virtual Class for Learning Algorithms
 */
class Learning{

public:
    /**
     * @brief  Class Constructor
     */
    Learning(){};

    /**
     * @brief Class Deconstructor
     */
    virtual ~Learning(){};

    /**
     * @brief Train Classifier
     * @param trainData Training data, size of {number of samples x number of features}
     * @param labels    Ground truth class labels, size of {1 x number of samples}
     */
    virtual void Train(Mat& trainData, const Mat& labels= Mat())=0;

    /**
     * @brief Predict class probabilities
     * @param testData          Test data,  size of {number of samples x number of features}
     * @param possibleLabels    argmax< class probabilities> for each test sample
     * @param labelProbs        Class probabilities for each test sample
     */
    virtual void Evaluate(Mat& testData, Mat& possibleLabels, Mat& labelProbs)=0;

    /**
     * @brief Save Classifier
     */
    virtual void SaveClassifier()=0;

    /**
     * @brief Load Classifier
     */
    virtual void LoadClassifier()=0;

    /**
     * @brief Clear classifier object
     */
    virtual void ClearClassifier()=0;

public:
    /**
     * @brief Return estimated training time
     * @return              Estimated time in miliseconds
     */
    int         getTrainTime();

    /**
     * @brief Path to the classifier file
     */
    string      trainFile;

    /**
     * @brief Name of the classifier
     */
    string      name_;

protected:
    /**
     * @brief Total number of classes
     */
    uint        nClass;

    /**
     * @brief Timer to estimate the time
     */
    Timer       timer;

    /**
     * @brief Training time in miliseconds
     */
    int         train_time;
};

/**
 * @brief The CvRTreesMultiClass class
 * This class has been taken from the following link:
 * http://stackoverflow.com/questions/10358964/using-opencv-random-forests-is-there-any-way-to-obtain-the-confidence-level-f
 * This function estimates the class probabilities given a feature vector.
 * Each tree votes for one class label.
 */
class CvRTreesMultiClass : public CvRTrees
{
    public:

    /**
     * @brief Predict unnormalized class probability for one test sample
     * @param sample                Test sample
     * @param out_votes             Unnormalized class probabilities for a given test sample
     * @param missing               Optional missing measurement mask of the sample (OpenCV doc)
     * @return                      Total number of trees in the forest
     */
    int predict_multi_class( const CvMat* sample,
                             int out_votes[],
                             const CvMat* missing = 0
                            ) const
        {
            if( nclasses > 0 ) // classification
            {
                int* votes = out_votes;
                memset( votes, 0, sizeof(*votes)*nclasses );
                for(int k = 0; k < ntrees; k++ )
                {
                    CvDTreeNode* predicted_node = trees[k]->predict( sample, missing );
                    int class_idx = predicted_node->class_idx;
                    CV_Assert( 0 <= class_idx && class_idx < nclasses );
                    ++votes[class_idx];
                }

            }
            else // regression
            {
                throw std::runtime_error("CvRTreesMultiClass predict_multi_class can only be used classification");
            }
            return ntrees;
        }
};

/**
 * @brief The RandomForest class
 */
class RandomForest: public Learning{

public:
    /**
     * @brief Class Constructor
     * @param clsFile       Path to the cassifier file
     * @param params        Structure for classifier parameters
     * @param numClasses    Total number of classes
     */
    RandomForest(string clsFile, CvRTParams params, unsigned int numClasses );

    void Train(Mat& trainData,const Mat& labels= Mat());
    void Evaluate(Mat& testData, Mat& possibleLabels,Mat& labelProbs);
    void SaveClassifier();
    void LoadClassifier();
    void ClearClassifier();

private:
    /**
     * @brief Classifier instance from CvRTreesMultiClass class
     */
    CvRTreesMultiClass      classifier;

    /**
     * @brief Parameters structure to store the Random Forest parameters
     */
    CvRTParams              parameters;
};

#endif // LEARNING_HPP
