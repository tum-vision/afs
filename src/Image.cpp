#ifndef IMAGE_CPP
#define IMAGE_CPP

#include <Image.hpp>

Image::Image(string impath, string depthpath){

    imName      = impath;

    BGRImage_   = imread( imName, CV_LOAD_IMAGE_COLOR );
    if( BGRImage_.empty() ){

        cout << "ERROR: Image not load : " << impath << endl;
        exit(-1);
    }
    depth       = imread(depthpath,CV_LOAD_IMAGE_GRAYSCALE);

    //    GaussianBlur(RGBImage_,RGBImage_,Size(3,3),0.5);
    LABImage_   = ImageProc::convertColor(BGRImage_,CV_BGR2Lab);

    ImageProc::getChannel(LABImage_,1).assignTo(L,CV_32F);
    ImageProc::getChannel(LABImage_,1).assignTo(L8U,CV_8U );
    ImageProc::getChannel(LABImage_,2).assignTo(a,CV_32F);
    ImageProc::getChannel(LABImage_,3).assignTo(b,CV_32F);

//    normalize(L, L, 0, 1, NORM_MINMAX, CV_32F);
//    normalize(a, a, 0, 1, NORM_MINMAX, CV_32F);
//    normalize(b, b, 0, 1, NORM_MINMAX, CV_32F);
}

Image::~Image(){

    this->BGRImage_.release();
    this->LABImage_.release();
    this->L.release();
    this->L8U.release();
    this->a.release();
    this->b.release();
    this->depth.release();
}



int Image::getHeight(){

    return BGRImage_.rows;
}


int Image::getWidth(){

    return BGRImage_.cols;
}

Mat Image::getImage(){

    return BGRImage_;
}

Mat Image::getRGB(){

    return ImageProc::convertColor(BGRImage_,CV_BGR2RGB);
}

bool Image::isEmpty(){

    return LABImage_.empty();
}
#endif // IMAGE_CPP
