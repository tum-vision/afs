#ifndef MODEL_HPP
#define MODEL_HPP

#include <common.hpp>
#include <Dataset.hpp>
#include <Feature.hpp>
#include <Learning.hpp>
#include <Optimization.hpp>

/**
 * @brief dummy time
 */
static int DummyTime = -1;

/**
 * @brief The Model class
 */
class Model{

public:

    /**
     * @brief Class Constructor
     */
    Model();

    /**
     * @brief Set model structure
     * @param dataset pointer to the dataset
     */
    void SetStructure(Dataset *dataset);

    /**
     * @brief Evaluate features
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     * @param eval_ftr_time estimated time per image in miliseconds
     */
    void EvaluateFeatures(vector<string>& imageList, int from=0, int to=-1, int& eval_ftr_time = DummyTime);

    /**
     * @brief Train potentials
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     * @param train_pt_time estimated time in miliseconds
     */
    void TrainPotentials(vector<string>& imageList,int from=0, int to=-1, int& train_pt_time = DummyTime);

    /**
     * @brief Evaluate potentials
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     * @param eval_pt_time  estimated time per image in miliseconds
     */
    void EvaluatePotentials(vector<string>& imageList,int from=0, int to=-1, int& eval_pt_time = DummyTime);

    /**
     * @brief Compute confusion matrix
     * @param imageList list of images
     * @param folder    path to the folder to save confusion matrix
     * @param confFileName  file name for confusion matrix
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     */
    void Confusion(vector<string>& imageList,string folder, string confFileName,int from=0, int to=-1);

    /**
     * @brief Solve optimization
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     * @param solving_time estimated time per image in miliseconds
     */
    void Solve(vector<string>& imageList,int from=0, int to=-1,  int& solving_time = DummyTime);

    /**
     * @brief Rank features with mrmr
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     * @param isRank    flag to rank features
     * @param ranking_time  estimated time to rank features
     */
    void RankFeatures(vector<string>& imageList, int from=0, int to=-1, bool isRank = true, int& ranking_time = DummyTime);

    /**
     * @brief Analyse features
     */
    void AnalyseFeatures();

    /**
     * @brief Select features
     * @param alpha alpha
     * @param beta  beta
     * @return  number of selected features
     */
    int  SelectFeatures(float alpha, float beta);

    /**
     * @brief Find optimal lambda for VariationalOptimization
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     */
    void FindOptimalLambda(vector<string>& imageList,int from=0, int to=-1);

    /**
     * @brief Save potential map for images
     * @param imageList list of images
     * @param from      start index
     * @param to        stop index, set to -1 to segment all images starting from "from"
     */
    void SavePotentialMap(vector<string>& imageList,int from=0, int to=-1);

    /**
     * @brief Activate fast computation
     * Load features instead of computing on the fly
     * @param isFast    flag to activate fast computation
     */
    void ActivateFastComputation(bool isFast = true);

    /**
     * @brief Print dataset info
     */
    void PrintDatasetInfo();

    /**
     * @brief get current performance
     * @return  current performance
     */
    Performance getPERFORMANCE();

private:

    /**
     * @brief flag to save confusion
     */
    bool SAVE_CONFUSION;

private:

    /**
     * @brief pointer to the current dataset
     */
    Dataset *dataset;

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
    Optimization   *optimizer;

    /**
     * @brief timer to estimate the time
     */
    Timer   timer;

    /**
     * @brief current performance
     */
    Performance _PERFORMANCE;

    /**
     * @brief current detection performance
     */
    Performance DET_PERFORMANCE;

    /**
     * @brief current segmentation performance
     */
    Performance SEG_PERFORMANCE;

    /**
     * @brief flag to activate fast computation
     */
    bool FAST_COMPUTATION;

};

#endif // MODEL_HPP
