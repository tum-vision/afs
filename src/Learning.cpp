
#include <Learning.hpp>

int Learning::getTrainTime(){

    return train_time;
}

RandomForest::RandomForest(string clsFile,
                           CvRTParams params,
                           unsigned int numClasses){

    trainFile               = clsFile;
    name_                   = "Random Forest";
    parameters              = params;
    nClass                  = numClasses;

}
void RandomForest::Train(Mat& trainData,const Mat& labels){

    int numOfFeatures = trainData.cols;
    Mat var_type = Mat(numOfFeatures +1 , 1, CV_8U );
    var_type.setTo(cv::Scalar(CV_VAR_NUMERICAL) );
    var_type.at<uchar>(numOfFeatures, 0) = CV_VAR_CATEGORICAL;

    timer.start();

    classifier.train(trainData,CV_ROW_SAMPLE,labels,Mat(),Mat(),var_type,Mat(),parameters);
    train_time = timer.stop("\t\telapsed time to train: ");
    cout<<                  "\t\tNumber of trees/variables: "<<classifier.get_tree_count()<<", ";
    cout<<trainData.cols<<endl;

    SaveClassifier();
    classifier.clear();

}
void RandomForest::Evaluate(Mat& testData, Mat& possibleLabels,Mat& labelProbs){


    int out_votes[nClass];

    for(int i= 0; i < testData.rows; i++){

        CvMat test_sample     = testData.row(i);
        float totalVotes = (float)classifier.predict_multi_class(&test_sample,out_votes);
        for (unsigned int j = 0; j < nClass; j++)
            labelProbs.at<float>(i,j) = (float)(out_votes[j]) / totalVotes;
    }
}
void RandomForest::SaveClassifier(){

    classifier.save(trainFile.c_str());
}
void RandomForest::LoadClassifier(){

    classifier.load(trainFile.c_str());
}
void RandomForest::ClearClassifier(){

    classifier.clear();
}
