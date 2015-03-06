#include "common.hpp"

// determine the edge detection function g
void edgeDetectionFct(cv::Mat I_Vec3b, cv::Mat &g)
{
    // convert image from 0 ... 255 to 0 ... 1
    cv::Mat I = cv::Mat(I_Vec3b.size(), CV_32FC3);
    I_Vec3b.convertTo(I, CV_32FC3, 1/255.0f, 0); // ATTENTION: if 32F type images have negative values included, they have to be converted!

    // compute the grayscale image and the edge detection function
    cv::Mat grayI(I.rows, I.cols, CV_32FC1);       // float 0 ... 1, 1 channel
    grayI.setTo(0);
    g.create(I.rows, I.cols, CV_32FC1);
    g.setTo(0);
    for(int r = 0; r < I.rows; r++){
        for(int c = 0; c < I.cols; c++){
            float channel_r = (float) I.at<cv::Vec3f>(r,c)[2];
            float channel_g = (float) I.at<cv::Vec3f>(r,c)[1];
            float channel_b = (float) I.at<cv::Vec3f>(r,c)[0];

            grayI.at<float>(r,c) = sqrtf( channel_r * channel_r   +  channel_g * channel_g   +   channel_b * channel_b );
        }
    }
    blur(grayI,grayI,Size(3,3));

    float norm_gradI;
    for(int r = 0; r < I.rows - 1; r++){
        for(int c = 0; c < I.cols - 1; c++){
            norm_gradI = sqrtf(
                        (grayI.at<float>(r  ,c+1)  -  grayI.at<float>(r,c) ) * ( grayI.at<float>(r,  c+1)  -  grayI.at<float>(r,c)  )
                      + (grayI.at<float>(r+1,c  )  -  grayI.at<float>(r,c) ) * ( grayI.at<float>(r+1,c  )  -  grayI.at<float>(r,c)  )
                          );

            g.at<float>(r,c) = exp(- 5 * norm_gradI );
        }
    }
}
