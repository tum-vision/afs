#include <common.hpp>
#include <ImageProc.hpp>



Mat ImageProc::getEdgeImage(Mat input){

    Mat gray, output;
    Mat dx_, dy_;

    cvtColor(input,gray,CV_RGB2GRAY,1);

    Sobel(gray,dx_,CV_16S,1,0,CV_SCHARR);
    Sobel(gray,dy_,CV_16S,0,1,3);

    convertScaleAbs(dx_,dx_);
    convertScaleAbs(dy_,dy_);

    addWeighted(dx_,0.5,dy_,0.5,0,output);

    return output;
}


Mat ImageProc::getIntegralImage(Mat input){

    Mat integralImage;
    integral(input,integralImage,CV_32F);
    return integralImage;
}

Mat ImageProc::getPaddedImg(Mat input,int extTopRows,int extBotRows,int extLeftCols, int extRightCols){

    Mat  output(input.rows+extTopRows + extBotRows, input.cols+extLeftCols+extRightCols, input.type(),Scalar(0,0,0));
    copyMakeBorder( input, output, extTopRows, extBotRows, extLeftCols, extRightCols, BORDER_REFLECT);

    return output;
}

Mat ImageProc::convertColor(Mat input,int code){

    Mat output;

    cvtColor(input,output,code);

    return output;
}

Mat ImageProc::getChannel(Mat input,unsigned char channel = 0){

    vector<Mat> output;
    split(input,output);

    return output[channel -1];
}
