
#include <Feature.hpp>

vector<string>feature_names_;

Feature::Feature(){

}

void Feature::setSubSample(int ss){

    subSample = ss;
}

int Feature::getSubSample(){

    return subSample;
}

int Feature::getFeatureDim(){

    return this->featureDim;
}

int Feature::getFullDim(){

    return this->featureSize * (this->allPatches.size() == 0 ? 1 : this->allPatches.size());
}

void Feature::setFeatureDim(int dim){

    featureDim = dim;
}

void Feature::saveFeature(string imName, Mat& features){

    foperator = fopen((ftFolder+imName+ext_).c_str(),"wb");
    if(foperator == NULL){

        cout<<"File not opened: "<< ftFolder+imName+ext_;
        exit(-1);
    }

    size_t size= fwrite(&features.rows,sizeof(int),1,foperator);
    if(size == 0){
        cout<<"Feature not saved: "<< ftFolder+imName+ext_;
        exit(-1);
    }
    fwrite(&features.cols,sizeof(int),1,foperator);
    fwrite(features.data,sizeof(float),features.rows*features.cols,foperator);
    fclose(foperator);

}

void Feature::loadFeature(string imName, Mat& features){

    int rows,cols;
    foperator = fopen((ftFolder+imName+ext_).c_str(),"rb");
    if(foperator == NULL){

        cout<<"File not opened: "<< ftFolder+imName+ext_;
        exit(-1);
    }

    size_t t = fread(&rows,sizeof(int),1,foperator);
    t        = fread(&cols,sizeof(int),1,foperator);

    features = Mat(rows,cols,CV_32F);
    t        =fread(features.data,sizeof(float),rows*cols,foperator);
    fclose(foperator);

}

void Feature::setRectSize(int height, int width){

    rHeight         = height;
    rWidth          = width;

    halfHeight      = height    / 2 ;
    halfWidth       = width     / 2 ;

    fourthOfHeight  = height    / 4 ;
    fourthOfWidth   = width     / 4 ;

    sixthOfHeight   = height    / 6 ;
    sixthOfWidth    = width     / 6 ;

}

HaarLikeFeatures::HaarLikeFeatures(vector<Size>& all_patches,string haarFolder, string ext,int numSubSample){

    name_       = "Haar";
    allPatches  = all_patches;
    ftFolder    = haarFolder;
    ext_        = ext;
    subSample   = numSubSample;
    featureSize = 18;
    featureDim  = featureSize*allPatches.size();

    isComputeFeature = Mat(1, featureDim, CV_8U, uchar(1));
    for(unsigned int i = 0; i < allPatches.size(); i ++ ){

        std::ostringstream ostr;
        ostr << all_patches[i].width;
        feature_names_.push_back("H_HE_"+ostr.str()+"_L");
        feature_names_.push_back("H_VE_"+ostr.str()+"_L");
        feature_names_.push_back("H_HL_"+ostr.str()+"_L");
        feature_names_.push_back("H_VL_"+ostr.str()+"_L");
        feature_names_.push_back("H_CS_"+ostr.str()+"_L");
        feature_names_.push_back("H_FS_"+ostr.str()+"_L");

        feature_names_.push_back("H_HE_"+ostr.str()+"_a");
        feature_names_.push_back("H_VE_"+ostr.str()+"_a");
        feature_names_.push_back("H_HL_"+ostr.str()+"_a");
        feature_names_.push_back("H_VL_"+ostr.str()+"_a");
        feature_names_.push_back("H_CS_"+ostr.str()+"_a");
        feature_names_.push_back("H_FS_"+ostr.str()+"_a");

        feature_names_.push_back("H_HE_"+ostr.str()+"_b");
        feature_names_.push_back("H_VE_"+ostr.str()+"_b");
        feature_names_.push_back("H_HL_"+ostr.str()+"_b");
        feature_names_.push_back("H_VL_"+ostr.str()+"_b");
        feature_names_.push_back("H_CS_"+ostr.str()+"_b");
        feature_names_.push_back("H_FS_"+ostr.str()+"_b");
    }
}


void HaarLikeFeatures::extractFeatures(Image* im, Mat& features){


    Mat im_L, im_a, im_b;
    int t, offset, indx, is;


    int subSampleRow = subSample>0 ? ((im->getHeight()+subSample-1) / subSample): im->getHeight();
    int subSampleCol = subSample>0 ? ((im->getWidth() +subSample-1) / subSample): im->getWidth();

    if(!subSample) subSample = 1;


    features = Mat(subSampleRow*subSampleCol,featureDim,CV_32F);

    indx = 0;
    is = 0; // for isComputeFeatures

    for(unsigned int i = 0; i < allPatches.size(); i ++ )
    {

        setRectSize(allPatches[i].height,allPatches[i].width);

        im_L = ImageProc::getIntegralImage(ImageProc::getPaddedImg(im->L,halfHeight, halfHeight, halfWidth, halfWidth));
        im_a = ImageProc::getIntegralImage(ImageProc::getPaddedImg(im->a,halfHeight, halfHeight, halfWidth, halfWidth));
        im_b = ImageProc::getIntegralImage(ImageProc::getPaddedImg(im->b,halfHeight, halfHeight, halfWidth, halfWidth));

        t  = 0;

        for(int row = 0; row< im->getHeight(); row= row+ subSample)
            for(int col= 0; col< im->getWidth(); col= col + subSample)
            {
                int rowShifted = row + halfHeight;
                int colShifted = row + halfWidth;
                offset = 0;

                if(isComputeFeature.at<uchar>(0,is   )) { horizontalEdge(im_L,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+1 )) { verticalEdge(im_L,rowShifted,colShifted,features.at<float>(t,indx+offset));     offset++;}
                if(isComputeFeature.at<uchar>(0,is+2 )) { horizontalLine(im_L,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+3 )) { verticalLine(im_L,rowShifted,colShifted,features.at<float>(t,indx+offset));     offset++;}
                if(isComputeFeature.at<uchar>(0,is+4 )) { centerSurround(im_L,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+5 )) { fourSquare(im_L,rowShifted,colShifted,features.at<float>(t,indx+offset));       offset++;}

                if(isComputeFeature.at<uchar>(0,is+6 )) { horizontalEdge(im_a,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+7 )) { verticalEdge(im_a,rowShifted,colShifted,features.at<float>(t,indx+offset));     offset++;}
                if(isComputeFeature.at<uchar>(0,is+8 )) { horizontalLine(im_a,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+9 )) { verticalLine(im_a,rowShifted,colShifted,features.at<float>(t,indx+offset));     offset++;}
                if(isComputeFeature.at<uchar>(0,is+10)) { centerSurround(im_a,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+11)) { fourSquare(im_a,rowShifted,colShifted,features.at<float>(t,indx+offset));       offset++;}

                if(isComputeFeature.at<uchar>(0,is+12)) { horizontalEdge(im_b,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+13)) { verticalEdge(im_b,rowShifted,colShifted,features.at<float>(t,indx+offset));     offset++;}
                if(isComputeFeature.at<uchar>(0,is+14)) { horizontalLine(im_b,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+15)) { verticalLine(im_b,rowShifted,colShifted,features.at<float>(t,indx+offset));     offset++;}
                if(isComputeFeature.at<uchar>(0,is+16)) { centerSurround(im_b,rowShifted,colShifted,features.at<float>(t,indx+offset));   offset++;}
                if(isComputeFeature.at<uchar>(0,is+17)) { fourSquare(im_b,rowShifted,colShifted,features.at<float>(t,indx+offset));       offset++;}

                t++;
            }

        is   += featureSize;
        indx += offset;
        im_L.release();
        im_a.release();
        im_b.release();
    }
}

void HaarLikeFeatures::horizontalEdge(Mat&im, int r, int c,float&fValue){


    weight1 = -1;
    weight0 =  1;

//    int A,B,C,D,E,F;
/*
    A---B
    |   |
    C---D
    |   |
    E---F
*/
    // White region is up, black region is down

    fValue = weight1 *
                    ((im.at<float>(r,c+halfWidth) + im.at<float>(r- halfHeight,c- halfWidth)) -
                    (im.at<float>(r-halfHeight,c+halfWidth) + im.at<float>(r,c- halfWidth))) +

            weight0 *
                    ((im.at<float>(r + halfHeight,c+ halfWidth) + im.at<float>(r,c-halfWidth)) -
                    (im.at<float>(r+ halfHeight,c -halfWidth) + im.at<float>(r,c + halfWidth)));

    fValue /= (rWidth * rHeight / 2);

}

void HaarLikeFeatures::verticalEdge(Mat&im, int r, int c, float& fValue){


    weight1 = -1;
    weight0 =  1;

    // White region is left, black region is right
    fValue = weight1 *
                     ((im.at<float>(r+halfHeight,c) + im.at<float>(r-halfHeight,c-halfWidth)) -
                     (im.at<float>(r- halfHeight,c) + im.at<float>(r+halfHeight,c-halfWidth)))   +

             weight0 *
                     ((im.at<float>(r+ halfHeight,c+ halfWidth) + im.at<float>(r-halfHeight,c)) -
                     (im.at<float>(r+halfHeight,c) + im.at<float>(r -halfHeight,c+ halfWidth)));


     fValue /= (rWidth * rHeight / 2);
}

void HaarLikeFeatures::horizontalLine(Mat&im, int r, int c, float &fValue){


    weight1 = -1;
    weight0 =  3;

    fValue  =  weight1 *
                       ((im.at<float>(r+halfHeight,c+halfWidth) + im.at<float>(r- halfHeight,c - halfWidth)) -
                       (im.at<float>(r+ halfHeight,c- halfWidth)  + im.at<float>(r- halfHeight,c+ halfWidth)))   +

               weight0 *
                       ((im.at<float>(r+sixthOfHeight,c+halfWidth) + im.at<float>(r- sixthOfHeight,c - halfWidth)) -
                       (im.at<float>(r+ sixthOfHeight,c- halfWidth)  + im.at<float>(r- sixthOfHeight,c+ halfWidth)));
     fValue  /= (rHeight * rWidth / 3);
}


void HaarLikeFeatures::verticalLine(Mat&im, int r, int c, float &fValue){


    weight1 = -1;
    weight0 =  3;

    fValue  =  weight1 *
                       ((im.at<float>(r+halfHeight,c+halfWidth) + im.at<float>(r- halfHeight,c - halfWidth)) -
                       (im.at<float>(r+ halfHeight,c- halfWidth)  + im.at<float>(r- halfHeight,c+ halfWidth)))   +

               weight0 *
                       ((im.at<float>(r+halfHeight,c+sixthOfWidth) + im.at<float>(r- halfHeight,c - sixthOfWidth)) -
                       (im.at<float>(r+ halfHeight,c- sixthOfWidth)  + im.at<float>(r- halfHeight,c+ sixthOfWidth)));

    fValue  /= (rHeight * rWidth / 3);
}

void HaarLikeFeatures::centerSurround(Mat&im, int r, int c, float &fValue){


    weight1 = -1;
    weight0 =  4;

    fValue  =  weight1 *
                       ((im.at<float>(r+ halfHeight,c+ halfWidth) + im.at<float>(r- halfHeight,c - halfWidth)) -
                       (im.at<float>(r+ halfHeight,c- halfWidth)  + im.at<float>(r- halfHeight,c+ halfWidth)))   +

                weight0 *
                        ((im.at<float>(r+ fourthOfHeight,c+ fourthOfWidth) + im.at<float>(r- fourthOfHeight,c - fourthOfWidth)) -
                        (im.at<float>(r+ fourthOfHeight,c- fourthOfWidth)  + im.at<float>(r- fourthOfHeight,c+ fourthOfWidth)));

    fValue  /= (rWidth*rHeight *3 / 4);
}

void HaarLikeFeatures::fourSquare(Mat&im, int r, int c, float &fValue){

    weight1 = -2;
    weight0 =  1; // weight for all here

    fValue  = weight1 *
                      ( // white region (top-left, down-right)
                        ((im.at<float>(r,c) + im.at<float>(r- halfHeight,c - halfWidth)) -
                        (im.at<float>(r ,c- halfWidth)  + im.at<float>(r- halfHeight,c))) +
                        ((im.at<float>(r,c) + im.at<float>(r+ halfHeight,c + halfWidth)) -
                        (im.at<float>(r+halfHeight ,c)  + im.at<float>(r,c+halfWidth)))
                      ) +
              weight0 *
                      ((im.at<float>(r+halfHeight,c+halfWidth) + im.at<float>(r- halfHeight,c - halfWidth)) -
                      (im.at<float>(r+halfHeight ,c- halfWidth)  + im.at<float>(r- halfHeight,c+halfWidth)));

    fValue  /= (rWidth*rHeight /2);
}

/* COLOUR FEATURES */


ColorFeatures::ColorFeatures(vector<Size>& all_patches,string colorFolder, string ext,int numSubSample){

    name_       = "Color";
    allPatches  = all_patches;
    ftFolder    = colorFolder;
    ext_        = ext;
    subSample   = numSubSample;

    srand (time(NULL));
    for(unsigned int i=0; i< allPatches.size(); i++){
        rndLocs.push_back(Point2d(rand() % allPatches[i].width - allPatches[i].width / 2, rand() % allPatches[i].height - allPatches[i].height / 2));
        rndRelatives.push_back(Point2d(rand() % allPatches[i].width - allPatches[i].width / 2  ,rand() % allPatches[i].height - allPatches[i].height / 2));
        rndRelatives.push_back(Point2d(rand() % allPatches[i].width - allPatches[i].width / 2  ,rand() % allPatches[i].height - allPatches[i].height / 2));

        // random Relative Patches
        Point2d size    = Point2d(rand() % (allPatches[i].width / 2 -1) +1, rand() % (allPatches[i].height / 2 -1) +1) ; // w - h
        Point2d xy      = Point2d(rand() % (allPatches[i].width / 2) - allPatches[i].width / 2, rand() % (allPatches[i].height / 2) - allPatches[i].height / 2) ; // w - h
        rndPatches.push_back(Rect(xy.x, xy.y, size.x, size.y));


        std::ostringstream ostr;
        ostr << all_patches[i].width;
        feature_names_.push_back("C_RC_"+ostr.str()+"_L");
        feature_names_.push_back("C_RP_"+ostr.str()+"_L");
        feature_names_.push_back("C_RC_"+ostr.str()+"_a");
        feature_names_.push_back("C_RP_"+ostr.str()+"_a");
        feature_names_.push_back("C_RC_"+ostr.str()+"_b");
        feature_names_.push_back("C_RP_"+ostr.str()+"_b");
    }

    featureSize = 6;
    featureDim  = featureSize*allPatches.size();
    isComputeFeature = Mat(1, featureDim, CV_8U, uchar(1));
}


void ColorFeatures::extractFeatures(Image* im, Mat& features){


    Mat im_L,im_a,im_b,integral_L,integral_a,integral_b;
    int t, offset, indx, is;

    int subSampleRow = subSample>0 ? ((im->getHeight()+subSample-1) / subSample): im->getHeight();
    int subSampleCol = subSample>0 ? ((im->getWidth() +subSample-1) / subSample): im->getWidth();

    if(!subSample) subSample = 1;


    features = Mat(subSampleRow*subSampleCol,featureDim,CV_32F);

    indx = 0;
    is = 0; // for isComputeFeatures

    for(unsigned int i = 0; i < allPatches.size(); i ++ )
    {
        setRectSize(allPatches[i].height,allPatches[i].width);

        im_L       = ImageProc::getPaddedImg(im->L,halfHeight, halfHeight, halfWidth, halfWidth);
        im_a       = ImageProc::getPaddedImg(im->a,halfHeight, halfHeight, halfWidth, halfWidth);
        im_b       = ImageProc::getPaddedImg(im->b,halfHeight, halfHeight, halfWidth, halfWidth);

        integral_L = ImageProc::getIntegralImage(im_L);
        integral_a = ImageProc::getIntegralImage(im_a);
        integral_b = ImageProc::getIntegralImage(im_b);


        t = 0 ;

        for(int row = 0; row< im->getHeight(); row= row+ subSample)
            for(int col= 0; col< im->getWidth(); col= col + subSample)
            {
                offset =0;
                int rowShifted = row + halfHeight;
                int colShifted = col + halfWidth;

                if(isComputeFeature.at<uchar>(0,is  )){ relativeColor(im_L,rowShifted,colShifted,i,features.at<float>(t,indx+offset));            offset++;}
                if(isComputeFeature.at<uchar>(0,is+1)){ relativePatch(integral_L,rowShifted,colShifted,i,features.at<float>(t,indx+offset));      offset++;}

                if(isComputeFeature.at<uchar>(0,is+2)){ relativeColor(im_a,rowShifted,colShifted,i,features.at<float>(t,indx+offset));            offset++;}
                if(isComputeFeature.at<uchar>(0,is+3)){ relativePatch(integral_a,rowShifted,colShifted,i,features.at<float>(t,indx+offset));      offset++;}

                if(isComputeFeature.at<uchar>(0,is+4)){ relativeColor(im_b,rowShifted,colShifted,i,features.at<float>(t,indx+offset));            offset++;}
                if(isComputeFeature.at<uchar>(0,is+5)){ relativePatch(integral_b,rowShifted,colShifted,i,features.at<float>(t,indx+offset));      offset++;}

                t++;
            }

        is   += featureSize;
        indx += offset;

        im_L.release();
        im_a.release();
        im_b.release();

        integral_a.release();
        integral_a.release();
        integral_b.release();
    }
}

void ColorFeatures::pixelColor(Mat& im, int r, int c, float &cValue){

    cValue  =  im.at<float>(r,c);
}

void ColorFeatures::relativeColor(Mat& im, int r, int c, int i, float &cValue){

    cValue  =  im.at<float>(r+ rndLocs[i].y, c+ rndLocs[i].x);
}

void ColorFeatures::relativeColorComp(Mat& im, int r, int c, int i, float &cValue){

    cValue  =   (float)(im.at<float>(r + rndRelatives[2*i].y,c + rndRelatives[2*i].x) <= im.at<float>(r + rndRelatives[2*i+1].y,c + rndRelatives[2*i+1].x));
}

void ColorFeatures::relativePatch(Mat&im, int r, int c, int i, float& fValue){

    Point2d start_ = Point2d(c + rndPatches[i].x, r + rndPatches[i].y);

    fValue = ((im.at<float>(start_.y + rndPatches[i].height, start_.x + rndPatches[i].width) + im.at<float>(start_.y, start_.x)) -
              (im.at<float>(start_.y, start_.x + rndPatches[i].width) + im.at<float>(start_.y + rndPatches[i].height, start_.x)));

    fValue /= (float)rndPatches[i].area();

    if(isnan(fValue))
        cout<<"NAN CLR_relative-patch: "<< r << " " << c<< " " << rndPatches[i].x << " "<< rndPatches[i].y <<" "<< rndPatches[i].width << " "<< rndPatches[i].height<<endl;

}

void ColorFeatures::horizontalColorEdge(Mat& im, int r, int c, float &cValue){

    weight1 = -1;
    weight0 =  1;


    cValue = weight1 *
                    ((im.at<float>(r,c+halfWidth) + im.at<float>(r- halfHeight,c- halfWidth)) -
                    (im.at<float>(r-halfHeight,c+halfWidth) + im.at<float>(r,c- halfWidth))) +

            weight0 *
                    ((im.at<float>(r + halfHeight,c+ halfWidth) + im.at<float>(r,c-halfWidth)) -
                    (im.at<float>(r+ halfHeight,c -halfWidth) + im.at<float>(r,c + halfWidth)));

    cValue /= (rWidth * rHeight / 2);

}
void ColorFeatures::verticalColorEdge(Mat& im, int r, int c, float &cValue){


    weight1 = -1;
    weight0 =  1;

    // White region is left, black region is right
    cValue = weight1 *
                     ((im.at<float>(r+halfHeight,c) + im.at<float>(r-halfHeight,c-halfWidth)) -
                     (im.at<float>(r- halfHeight,c) + im.at<float>(r+halfHeight,c-halfWidth)))   +

             weight0 *
                     ((im.at<float>(r+ halfHeight,c+ halfWidth) + im.at<float>(r-halfHeight,c)) -
                     (im.at<float>(r+halfHeight,c) + im.at<float>(r -halfHeight,c+ halfWidth)));


     cValue /= (rWidth * rHeight / 2);
}
void ColorFeatures::centerSurround(Mat&im, int r, int c, float &fValue){


    weight1 = -1;
    weight0 =  4;

    fValue  =  weight1 *
                       ((im.at<float>(r+ halfHeight,c+ halfWidth) + im.at<float>(r- halfHeight,c - halfWidth)) -
                       (im.at<float>(r+ halfHeight,c- halfWidth)  + im.at<float>(r- halfHeight,c+ halfWidth)))   +

                weight0 *
                        ((im.at<float>(r+ fourthOfHeight,c+ fourthOfWidth) + im.at<float>(r- fourthOfHeight,c - fourthOfWidth)) -
                        (im.at<float>(r+ fourthOfHeight,c- fourthOfWidth)  + im.at<float>(r- fourthOfHeight,c+ fourthOfWidth)));

    fValue  /= (rWidth*rHeight *3 / 4);
}

LocationFeatures::LocationFeatures(string LocFolder, string ext, int numSubSample){

    name_       = "Location";
    ftFolder    = LocFolder;
    ext_        = ext;
    subSample   = numSubSample;
    featureSize = 2;
    featureDim  = featureSize;

    isComputeFeature = Mat(1, featureDim, CV_8U, uchar(1));

    feature_names_.push_back("L_x");
    feature_names_.push_back("L_y");

}

void LocationFeatures::extractFeatures(Image* im, Mat& features){

    int t;

    int subSampleRow = subSample>0 ? ((im->getHeight()+subSample-1) / subSample): im->getHeight();
    int subSampleCol = subSample>0 ? ((im->getWidth() +subSample-1) / subSample): im->getWidth();

    if(!subSample) subSample = 1;


    features = Mat(subSampleRow*subSampleCol,featureDim,CV_32F);

    t = 0 ;
    for(int row = 0; row< im->getHeight(); row= row+ subSample)
        for(int col= 0; col< im->getWidth(); col= col + subSample)
        {
            if(isComputeFeature.at<uchar>(0,0)) features.at<float>(t,1) = (float)col / (float)im->getWidth() ;
            if(isComputeFeature.at<uchar>(0,1)) features.at<float>(t,0) = (float)row / (float)im->getHeight();
            t++;
        }
}

DepthFeatures::DepthFeatures(vector<Size>& all_patches, string depthFolder, string ext, int numSubSample){

    name_         = "Depth";
    allPatches    = all_patches;
    ftFolder      = depthFolder;
    ext_          = ext;
    subSample     = numSubSample;
    featureSize   = 3;
    featureDim    = featureSize*allPatches.size();

    isComputeFeature = Mat(1, featureDim, CV_8U, uchar(1));
    for(unsigned int i=0; i< allPatches.size(); i++){

        rndRelatives.push_back(Point2d(rand() % allPatches[i].width - allPatches[i].width / 2  ,rand() % allPatches[i].height - allPatches[i].height / 2));
        rndRelatives.push_back(Point2d(rand() % allPatches[i].width - allPatches[i].width / 2  ,rand() % allPatches[i].height - allPatches[i].height / 2));


        std::ostringstream ostr;
        ostr << all_patches[i].width;
        feature_names_.push_back("D_RD_"+ostr.str()+"_D");
        feature_names_.push_back("D_PH_"+ostr.str()+"_D");
        feature_names_.push_back("D_DC_"+ostr.str()+"_D");

    }
}

void DepthFeatures::extractFeatures(Image *im, Mat &features){

    Mat paddedImage;
    if(im->depth.empty()){
        cout<<"Depth Feature: Depth image not loaded!"<<endl;
        exit(-1);
    }
    int t, offset, indx, is;

    int subSampleRow = subSample>0 ? ((im->getHeight()+subSample-1) / subSample): im->getHeight();
    int subSampleCol = subSample>0 ? ((im->getWidth() +subSample-1) / subSample): im->getWidth();

    if(!subSample) subSample = 1;


    features    = Mat(subSampleRow*subSampleCol,featureDim,CV_32F);
    reduce(im->depth, maxIncolumn, 0, CV_REDUCE_MAX);

    indx = 0;
    is = 0; // for isComputeFeatures

    for(unsigned int i = 0; i < allPatches.size(); i ++ )
    {
        setRectSize(allPatches[i].height,allPatches[i].width);
        paddedImage       = ImageProc::getPaddedImg(im->depth,halfHeight, halfHeight, halfWidth, halfWidth);

        t = 0 ;

        for(int row = 0; row< im->getHeight(); row= row+ subSample)
            for(int col= 0; col< im->getWidth(); col= col + subSample)
            {
                offset =0;
                int rowShifted = row + halfHeight;
                int colShifted = col + halfWidth;

                if(isComputeFeature.at<uchar>(0,is  )){ relativeDepth(paddedImage, rowShifted, colShifted, features.at<float>(t,indx+offset));    offset++;}
                if(isComputeFeature.at<uchar>(0,is+1)){ pointHeight(paddedImage, rowShifted, colShifted, features.at<float>(t,indx+offset));      offset++;}
                if(isComputeFeature.at<uchar>(0,is+2)){ depthComp(paddedImage, rowShifted, colShifted, i, features.at<float>(t,indx+offset));     offset++;}

                t++;
            }

        is   += featureSize;
        indx += offset;

        paddedImage.release();
   }
}

void DepthFeatures::relativeDepth(Mat &im, int r, int c, float &fValue){

    fValue  =   im.at<uchar>(r,c) / ((float)maxIncolumn.at<uchar>(0, c) == 0 ? 1 : (float)maxIncolumn.at<uchar>(0, c));
}

void DepthFeatures::pointHeight(Mat &im, int r, int c, float &fValue){

    fValue  =  - (float)im.at<uchar>(r,c) * (r- halfHeight);
}

void DepthFeatures::depthComp(Mat &im, int r, int c, int i, float &fValue){

    float pixelDepth  =  ((float)im.at<uchar>(r,c)==0 ? 1 : (float)im.at<uchar>(r,c));

    fValue  =  (float) im.at<uchar>(int(r + rndRelatives[2*i+1].y / pixelDepth), int(c + rndRelatives[2*i+1].x / pixelDepth)) -
               (float) im.at<uchar>(int(r + rndRelatives[2*i].y / pixelDepth), int(c + rndRelatives[2*i].x / pixelDepth));
}

TextonFeatures::TextonFeatures(string TextonFolder, string ext, int numSubSample, double bandWidth){

    name_               = "Texton";
    ftFolder            = TextonFolder;
    ext_                = ext;
    subSample           = numSubSample;
    kappa               = bandWidth;
    featureSize         = 17; // numTextons
    featureDim          = featureSize;

    isComputeFeature = Mat(1, featureDim, CV_8U, uchar(1));

    feature_names_.push_back("T_G_3_L");
    feature_names_.push_back("T_G_3_a");
    feature_names_.push_back("T_G_3_b");
    feature_names_.push_back("T_G_5_L");
    feature_names_.push_back("T_G_5_a");
    feature_names_.push_back("T_G_5_b");
    feature_names_.push_back("T_G_9_L");
    feature_names_.push_back("T_G_9_a");
    feature_names_.push_back("T_G_9_b");
    feature_names_.push_back("T_DoGx_5x13_L");
    feature_names_.push_back("T_DoGy_13x5_L");
    feature_names_.push_back("T_DoGx_9x25_L");
    feature_names_.push_back("T_DoGy_25x9_L");
    feature_names_.push_back("T_LoG_3_L");
    feature_names_.push_back("T_LoG_5_L");
    feature_names_.push_back("T_LoG_9_L");
    feature_names_.push_back("T_LoG_17_L");


}

void TextonFeatures::extractFeatures(Image *im, Mat &filterResponses){

    const int nFilters = featureDim;
    vector<Mat> response(nFilters);
    Mat image32f, filteredImage;
    vector<Mat> channels;

    int t;

    int subSampleRow = subSample>0 ? ((im->getHeight()+subSample-1) / subSample): im->getHeight();
    int subSampleCol = subSample>0 ? ((im->getWidth() +subSample-1) / subSample): im->getWidth();

//    subSampleRow = im->getHeight();
//    subSampleCol = im->getWidth();
    int tmp      = subSample;

    if(!subSample) subSample = 1;

    filterResponses    = Mat(subSampleRow*subSampleCol,featureDim,CV_32F,Scalar(0.0f));

    // TEXTON FILTERS //

    for (int i = 0; i < featureDim; i++)
            response[i] = Mat(im->getHeight(), im->getWidth() , CV_32FC1);

    im->LABImage_.convertTo(image32f,CV_32F,1.0 / 255.0);
    split(image32f, channels);

    int is = 0; // used as real feature address in a full vector of 17 texton elements
    int index_response = 0;

    // gaussian filter on all color channels
    for (double sigma = 1.0; sigma <= 4.0; sigma *= 2.0) {
        const int h = 2 * (int)(kappa * sigma) + 1;
        if(isComputeFeature.at<uchar>(0,is++)){

//            cout << is-1 <<" "<< feature_names_[is-1] << endl;

            GaussianBlur(channels[0], channels[0], Size(h, h), 0);
            response[index_response++] = channels[0].clone();
        }
        if(isComputeFeature.at<uchar>(0,is++)){

            //            cout << is-1 <<" "<< feature_names_[is-1] << endl;

            GaussianBlur(channels[1], channels[1], Size(h, h), 0);
            response[index_response++] = channels[1].clone();
        }
        if(isComputeFeature.at<uchar>(0,is++)){

            //            cout << is-1 <<" "<< feature_names_[is-1] << endl;

            GaussianBlur(channels[2], channels[1], Size(h, h), 0);
            response[index_response++] = channels[1].clone();
        }
    }

    // derivatives of gaussians on just greyscale image
    for (double sigma = 2.0; sigma <= 4.0; sigma *= 2.0) {

        if(isComputeFeature.at<uchar>(0,is++)){

            //            cout << is-1 <<" "<< feature_names_[is-1] << endl;

            // x-direction
            Sobel(channels[0], response[index_response++], CV_32F, 1, 0, 1);
            GaussianBlur(response[index_response - 1], response[index_response - 1],
                Size(2 * (int)(kappa * sigma) + 1, 2 * (int)(3.0 * kappa * sigma) + 1), 0);
        }
        if(isComputeFeature.at<uchar>(0,is++)){

            //            cout << is-1 <<" "<< feature_names_[is-1] << endl;

            // y-direction
            Sobel(channels[0], response[index_response++], CV_32F, 0, 1, 1);
            GaussianBlur(response[index_response - 1], response[index_response - 1],
                Size(2 * (int)(3.0 * kappa * sigma) + 1, 2 * (int)(kappa * sigma) + 1), 0);
        }
    }

    // laplacian of gaussian on just greyscale image
    Mat tmpImg(channels[0].rows, channels[0].cols, CV_32FC1);
    for (double sigma = 1.0; sigma <= 8.0; sigma *= 2.0) {
        if(isComputeFeature.at<uchar>(0,is++)){

            //            cout << is-1 <<" "<< feature_names_[is-1] << endl;

            const int h = 2 * (int)(kappa * sigma) + 1;
            GaussianBlur(channels[0], tmpImg, Size(h, h), 0);
            Laplacian(tmpImg, response[index_response++], CV_32F, 3);
        }
    }

    for(int i = 0; i < nFilters; i ++ )
    {
        t = 0 ;
        for(int row = 0; row< im->getHeight(); row= row+ subSample)
            for(int col= 0; col< im->getWidth(); col= col + subSample)
            {
                filterResponses.at<float>(t++,i) = response[i].at<float>(row,col);
            }
        response[i].release();
    }

    filteredImage.release();
    image32f.release();
    tmpImg.release();

    subSample   =   tmp;
}
