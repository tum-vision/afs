#include "Optimization.hpp"
#include "Dataset.hpp"

VariationalOptimization::VariationalOptimization(){

}

int VariationalOptimization::Solve(Dataset* dataset,vector<string>& imageList,int from, int to, bool printTime){

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);
    totalTime = 0;

    for(int i=from; i< to; i++ )
    {
        // load image
        cv::Mat I = cv::imread(dataset->imageFolder+imageList[i]+dataset->imageFileExt);
        if (I.empty()){
            cout << "\nCannot load the image: " << dataset->imageFolder+imageList[i]+dataset->imageFileExt << endl;
            exit(1);
        }

        // determine the edge detection function g(x)
        cv::Mat g;
        edgeDetectionFct(I, g);

        // initialize the dataterm
        int dim_size[3] = {dataset->n, I.rows, I.cols};
        cv::Mat dataterm(3, dim_size, CV_32FC1);
        dataterm.setTo(0);

        // load dataterm
        dataset->potentials[0]->loadPotential(imageList[i],dataterm);

        //===========================================================================
        // computation of segmentation
        // initialize Matrix including all indicator functions u: allocate memory for regions of segmentation & set first region to 1
        cv::Mat Mat_u(3, dim_size, CV_32FC1);   // float,   1 channel, dim_size[3] = {n, I.rows, I.cols}
        Mat_u.setTo(1);

        // GPU: compute the segmentation & count the time
        double time_segm;
        int steps = dataset->maxSteps;
        call_segmentation((float*) dataterm.data, (float*) g.data, (float*) Mat_u.data, I.cols, I.rows, dataset->n, dataset->lambda, steps, time_segm);
        totalTime += (time_segm * 1000.0);

        //===========================================================================
        // binarize u
        cv::Mat u_binary(I.rows, I.cols, CV_8UC1);
        call_binarization((float*) Mat_u.data, u_binary.data, I.cols, I.rows, dataset->n);

        //===========================================================================
        // get resulting regions
        cv::Mat regions;
        u_binary  += 1;
        dataset->Label2RGB(u_binary,regions);

        imwrite(dataset->resultFolder+imageList[i]+dataset->grFileExt,regions,compression_params);

        destroyAllWindows();
    }
    if(printTime){

        timer.msg(totalTime,                    "\t\telapsed time to segment ~ ");
        timer.msg(totalTime / imageList.size(), "\t\t              per image ~ ");
    }

    return totalTime / imageList.size();
}
