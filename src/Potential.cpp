#include "Potential.hpp"
#include "Dataset.hpp"

void Potential::savePotential(string imName, Mat& potentials, int classNo){

    Mat doublePots;
    foperator = fopen((folder+imName+ext).c_str(),"wb");
    if(foperator == NULL){

        cout<<"File not opened: "<< folder+imName+ext;
        exit(-1);
    }

    potentials.assignTo(doublePots,CV_64F);

    int cols = int(doublePots.cols/classNo);
    fwrite(&cols,sizeof(int),1,foperator);
    fwrite(&doublePots.rows,sizeof(int),1,foperator);
    fwrite(&classNo,sizeof(int),1,foperator);
    fwrite(doublePots.data,sizeof(double),doublePots.rows*doublePots.cols,foperator);
    fclose(foperator);

}

void Potential::loadPotential(string imName, Mat& potentials){

    int rows,cols, classNo;
    foperator = fopen((folder+imName+ext).c_str(),"rb");
    if(foperator == NULL){

        cout<<"File not opened: "<< folder+imName+ext;
        exit(-1);
    }


    size_t t = fread(&cols,sizeof(int),1,foperator);
    t        = fread(&rows,sizeof(int),1,foperator);
    t        = fread(&classNo,sizeof(int),1,foperator);

    Mat doublePots = Mat(rows, cols * classNo, CV_64F);
    t        = fread(doublePots.data,sizeof(double),rows*cols*classNo,foperator);
    fclose(foperator);

    int tmpC;
    for(int cl = 0; cl < classNo; cl++){
        for(int r = 0; r < rows; r++){
            tmpC = 0;
            for(int c = cl * cols; c < cl * cols + cols; c++){
                potentials.at<float>(cl,r,tmpC++) = -log((float) doublePots.at<double>(r,c));
            }
        }
    }

}

DenseUnaryPixelPotential::DenseUnaryPixelPotential(string dir, string fileExt){

    name_   =   "Dense Unary Pixel Potential";
    folder  =   dir;
    ext     =   fileExt;
}

int DenseUnaryPixelPotential::Train(Dataset* dataset,vector<string>& imageList,int from, int to, bool FAST_COMPUTATION){

    Mat trainData, sampleData, nonZeroLabels;
    Mat labelDensity(1,dataset->numClasses,CV_32F,Scalar(0));
    Mat logDensity(1,dataset->numClasses,CV_32F,Scalar(0));

    int totalFeatureType = 0;
    dataset->trainEntropy = 0;
    for(int f=0; f< dataset->totalFeatureType; f++ ){

        totalFeatureType += (dataset->features[f]->getFeatureDim() > 0);
    }

    for(int i=from; i< to; i++ ){

        vector<Mat> FeatureSet(totalFeatureType);
        Mat labels;
//        cout << dataset->grFolder+imageList[i]+dataset->grFileExt <<endl;
        Image *groundTruth = new Image(dataset->grFolder+imageList[i]+dataset->grFileExt);
        dataset->RGB2Label(groundTruth->getRGB(),labels);
        delete groundTruth;

        if(sum(labels).val[0] == 0){

            cout<<"\t\t\tWARNING: NONLABELED IMAGE: "<< imageList[i]<<endl;
        }
        else{

            int ftr_indx = 0;
            for(int f=0; f< dataset->totalFeatureType; f++ ){

                if(dataset->features[f]->getFeatureDim() !=0){

                    Mat ftr, subFeatureSet;
                    dataset->features[f]->loadFeature(imageList[i],ftr);

                    int subSample = dataset->features[f]->getSubSample();
                    int index = 0; // Get only non zero labeled features


                    if(FAST_COMPUTATION){

                        subFeatureSet= Mat(ftr.rows, dataset->features[f]->getFeatureDim(), ftr.type(), Scalar::all(0.0));
                        int internalIndex = 0;

                        for(int sF = 0; sF < dataset->features[f]->getFullDim(); sF++){

//                            cout << sF << " "<< (int)dataset->features[f]->isComputeFeature.at<uchar>(0,sF) << endl;
                            if((int)dataset->features[f]->isComputeFeature.at<uchar>(0,sF)){

                                ftr.col(sF).copyTo(subFeatureSet.col(internalIndex));
                                internalIndex++;
                            }
                        }
                        ftr.release();
                        ftr = Mat(subFeatureSet);
                        subFeatureSet.copyTo( ftr );
                        subFeatureSet.release();
                    }

                     for(int r=0; r< labels.rows; r= r+ subSample)
                        for(int c=0; c< labels.cols; c = c+ subSample){

                            if(labels.at<float>(r,c) != 0.0){

                                if(ftr_indx == 0){ // get labels once
                                    nonZeroLabels.push_back(labels.at<float>(r,c));
                                }
                                FeatureSet[ftr_indx].push_back(ftr.row(index));
                                labelDensity.at<float>(0,(int)labels.at<float>(r,c) -1) += 1;
                            }
                            index++;
                        }

                    ftr_indx++ ;
                    ftr.release();
                }
            }
            hconcat(FeatureSet,sampleData); // Features for one image
            trainData.push_back(sampleData);
            for(int f=0; f< totalFeatureType; f++ ) FeatureSet[f].release();
            FeatureSet.clear();
            sampleData.release();
            labels.release();
        }
    }

    labelDensity /= (float)sum(labelDensity).val[0];
    cv::log(labelDensity, logDensity);

    dataset->trainEntropy = - sum(labelDensity.mul(logDensity / std::log(dataset->numClasses))).val[0];

    cout<<"\t\ttraining Feature "<<this->name_<<" with "<<this->learning->name_<<" learner...\n\t\t\tClass Density ";
    for(unsigned int cl=0; cl < dataset->numClasses;cl++) printf("%.3f ", labelDensity.at<float>(0,cl));
    cout<<" Entropy : "<<dataset->trainEntropy<<endl;
    if(dataset->class_weights){
        cout<<"\t\t\tClass Weights ";
        if(dataset->class_weights !=0) for(unsigned int cl=0; cl < dataset->numClasses;cl++) printf("%.3f ", dataset->class_weights[cl]);
        cout<<endl;
    }

    this->learning->Train(trainData, nonZeroLabels);

    trainData.release();
    nonZeroLabels.release();

    return this->learning->getTrainTime();
}

int DenseUnaryPixelPotential::Evaluate(Dataset* dataset, vector<string>& imageList, int from, int to, bool FAST_COMPUTATION){

    int totalTime = 0;
    dataset->testEntropy = 0;
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);


    cout<<"\t\tevaluating "<<this->name_<<" with "<<this->learning->name_<<endl;

    Mat rgb, possibleLabels, dummy;
    Mat potentialImage;
    vector<Mat> pots(dataset->numClasses);
    Image *image;
    vector<int> subSampleArray(dataset->totalFeatureType); // keep subsamples to set them back
    int totalFeatureType = 0;

    this->learning->LoadClassifier();

    for(int f=0; f< dataset->totalFeatureType; f++ ){
        totalFeatureType += (dataset->features[f]->getFeatureDim() > 0);
    }

    for(int f=0; f< dataset->totalFeatureType; f++ ){
        subSampleArray[f] = dataset->features[f]->getSubSample();
        dataset->features[f]->setSubSample(0);
    }

    for(int i=from; i < to; i++){

        vector<Mat> FeatureSet(totalFeatureType);
        image              = new Image(dataset->imageFolder+imageList[i]+dataset->imageFileExt,dataset->depthFolder+imageList[i]+dataset->depthFileExt);
        possibleLabels     = Mat(image->getHeight(),image->getWidth(),CV_8U);

        if( image->isEmpty()){
            cout<<"MODEL: Error loading image : "<< dataset->imageFolder+imageList[i]+dataset->imageFileExt<<endl;
            exit(-1);
        }

//        cout<<"\t\t"<<i+1<<". image: "<< dataset->imageFolder+imageList[i]+dataset->imageFileExt<<endl;
        Mat fts, locFtr, labels, feature_class_probs, location_class_probs;

        timer.start();

        int ftr_indx = 0;
        for(int f=0; f< dataset->totalFeatureType; f++ ){

            if(dataset->features[f]->getFeatureDim() != 0){

                Mat singleFeature,indexResult, result, subFeatureSet;

                if(FAST_COMPUTATION){

                    dataset->features[f]->loadFeature(imageList[i],singleFeature);

                    subFeatureSet= Mat(singleFeature.rows, dataset->features[f]->getFeatureDim(), singleFeature.type(), Scalar::all(0.0));
                    int internalIndex = 0;

                    for(int sF = 0; sF < dataset->features[f]->getFullDim(); sF++){

                        if((int)dataset->features[f]->isComputeFeature.at<uchar>(0,sF)){

                            singleFeature.col(sF).copyTo(subFeatureSet.col(internalIndex));
                            internalIndex++;
                        }
                    }
                    singleFeature.release();
                    singleFeature = Mat(subFeatureSet);
                    subFeatureSet.copyTo( singleFeature );
                    subFeatureSet.release();
                }
                else
                    dataset->features[f]->extractFeatures(image,singleFeature);

                FeatureSet[ftr_indx] = singleFeature.clone();

                ftr_indx++;
                singleFeature.empty();
            }

        }
        hconcat(FeatureSet,fts);

//        for(int f=0, ftr_indx = 0; f< dataset->totalFeatureType; f++ ){

//            if(dataset->features[f]->getFeatureDim() !=0){

//                Mat ftr;
//                dataset->features[f]->loadFeature(imageList[i],ftr);
//                FeatureSet[ftr_indx] = ftr.clone();

//                ftr_indx++ ;
//                ftr.empty();
//            }
//        }
//        hconcat(FeatureSet,fts); // Features for one image

        feature_class_probs = Mat(fts.rows,dataset->numClasses,CV_32F);

        this->learning->Evaluate(fts,dummy,feature_class_probs);

        totalTime += timer.stop();

        for(int cNo=0; cNo< (int)dataset->numClasses; cNo++){
            Mat tmp     = feature_class_probs.col(cNo).clone();
            tmp.reshape(1,image->getHeight()).assignTo(pots[cNo]);
        }

        hconcat(pots,potentialImage);

        this->savePotential(imageList[i],potentialImage,dataset->numClasses);

        if(dataset->saveDetections){

            Point maxLoc;
            Mat labels = Mat(fts.rows,1,CV_8U);
            for(int p= 0; p < fts.rows; p++){
                minMaxLoc(feature_class_probs.row(p),NULL,NULL,NULL,&maxLoc);
                labels.at<unsigned char>(p,0) = (unsigned char)(maxLoc.x+1);
            }

             labels.reshape(1,image->getHeight()).assignTo(possibleLabels,CV_8U);

            dataset->Label2RGB(possibleLabels,rgb);
            imwrite(dataset->detectionFolder+imageList[i]+dataset->grFileExt,rgb,compression_params);
            labels.release();
        }

        {
            Mat logProb, entropyMap, entropyImage8U, JETImage;
            Mat entropyImage(potentialImage.rows, potentialImage.cols / dataset->numClasses, CV_32F, Scalar::all(0));


            cv::log(potentialImage, logProb);
            entropyMap  =  - potentialImage.mul( logProb / std::log(dataset->numClasses));
            for(int cl=0 ; cl< dataset->numClasses; cl++){
                entropyImage += entropyMap(Range::all(), Range(cl* entropyImage.cols, cl* entropyImage.cols + entropyImage.cols));
            }

            dataset->testEntropy += sum(entropyImage).val[0] / (entropyImage.rows * entropyImage.cols);
            entropyImage = 1- entropyImage;
            entropyImage *= 255;
            entropyImage.assignTo(entropyImage8U,CV_8U);
            applyColorMap(entropyImage8U,JETImage,COLORMAP_JET);
//            imwrite("eTrims/Result/"+imageList[i]+"Entropy.png",JETImage,compression_params);

            if(dataset->showEntropy){

                namedWindow("Entropy of each class in image: "+imageList[i],CV_WINDOW_NORMAL);
                resizeWindow("Entropy of each class in image: "+imageList[i],entropyMap.cols/2,entropyMap.rows);

                imshow("Entropy of each class in image: "+imageList[i],entropyMap);
                imshow("Entropy image: "+imageList[i], JETImage);

                waitKey();
                destroyAllWindows();
            }
            logProb.release();
            entropyMap.release();
            entropyImage.release();
            JETImage.release();
        }

        if(dataset->savePotentials){

            Mat potentialImage8U,JETImage; // Warm-hot colormapping
            potentialImage *= 255;
            potentialImage.assignTo(potentialImage8U,CV_8U);
            applyColorMap(potentialImage8U,JETImage,COLORMAP_JET);
            imwrite(dataset->pMapFolder+imageList[i]+".png",JETImage,compression_params);

//            namedWindow("Potentials of image:"+imageList[i],CV_WINDOW_NORMAL);
//            imshow("Potentials of image:"+imageList[i],JETImage);
//            resizeWindow("Potentials of image:"+imageList[i],JETImage.cols/2,JETImage.rows);
//            waitKey();
//            destroyAllWindows();
            potentialImage8U.release();
            JETImage.release();
        }

        delete image;
        locFtr.release();
        fts.release();
        rgb.release();
        possibleLabels.release();
        feature_class_probs.release();
        location_class_probs.release();
        for(int f=0; f< totalFeatureType; f++ ) FeatureSet[f].release();
        FeatureSet.clear();
    }

    for(int f=0; f< dataset->totalFeatureType; f++ ) dataset->features[f]->setSubSample(subSampleArray[f]);

    timer.msg(totalTime,                    "\t\telapsed time to evaluate ~ ");
    timer.msg(totalTime / imageList.size(), "\t\t               per image ~ ");

    dataset->testEntropy /= imageList.size();
    cout<<"Detection Entropy: "<<setprecision(2)<<dataset->testEntropy<<endl;

    subSampleArray.clear();
    this->learning->ClearClassifier();

    return totalTime / (float)imageList.size();
}

