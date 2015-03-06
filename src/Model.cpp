
#include <Model.hpp>

Model::Model(){

}

void Model::SetStructure(Dataset *dataset){


    this->dataset     = dataset;
    this->features    = dataset->features;
    this->potentials  = dataset->potentials;
    this->optimizer   = dataset->optimizer;
    SAVE_CONFUSION    = true;
    FAST_COMPUTATION  = false;

}

void Model::PrintDatasetInfo(){

    cout<<"Dataset Name/Size      :"<< dataset->name_ <<"/"<< dataset->allImages.size() << endl;
    cout<<"Train image size       :"<< dataset->trainImgs.size()<< endl;
    cout<<"Validation image size  :"<< dataset->validationImgs.size()<< endl;
    cout<<"Test image size        :"<< dataset->testImgs.size() << endl;

}

Performance Model::getPERFORMANCE(){

    return _PERFORMANCE;
}

void Model::SavePotentialMap(vector<string>& imageList,int from, int to){

    if(to== -1) to = imageList.size();

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    string Pfolder = dataset->name_+"/Result/PotentialMaps/"+dataset->potentials[0]->learning->name_+"/";
    dataset->createDir(Pfolder);

    for(int i= from; i< to; i++ ){

        Mat potentialImage, potentialImage8U, labels, detectionImage8U, JETImage;;
        potentials[0]->loadPotential(imageList[i], potentialImage);

        // Warm-hot colormapping
        potentialImage *= 255;
        potentialImage.assignTo(potentialImage8U,CV_8U);
        applyColorMap(potentialImage8U,JETImage,COLORMAP_JET);
        imwrite(Pfolder+imageList[i]+".png",JETImage,compression_params);
        potentialImage8U.release();
        JETImage.release();
    }
}

void Model::EvaluateFeatures(vector<string>&imageList, int from, int to, int &eval_ftr_time){


    if(to== -1) to = imageList.size();

    int numFeatureDims = 0;
    int totalFeatureType = 0;
    int totalTime = 0.0;
    Timer feature_evaluation_time;

    for(int f=0; f< dataset->totalFeatureType; f++)
        if(features[f]->getFeatureDim() !=0 ){
            totalFeatureType++;
            numFeatureDims += features[f]->getFeatureDim();
        }

    cout<<"Evaluating Features... Total #FeatureType: "<< totalFeatureType <<endl;

    for(int f=0; f< dataset->totalFeatureType; f++ ){

        if(features[f]->getFeatureDim() !=0 ){ // Dont compute if the feature set is empty !!

            cout<<"Evaluating " << features[f]->name_ <<" features...\tSize = 1x"<< features[f]->getFeatureDim()<<""<<endl;

            for(int i=from; i< to; i++ )
            {

                Image *im = new Image(dataset->imageFolder+imageList[i]+dataset->imageFileExt,dataset->depthFolder+imageList[i]+dataset->depthFileExt);;

                if( im->isEmpty()){
                    cout<<"MODEL: Error loading image : "<< dataset->imageFolder+imageList[i]+dataset->imageFileExt<<endl;
                    exit(-1);
                }


                Mat fts, indexResult, result;

                feature_evaluation_time.start();

                features[f]->extractFeatures(im,fts);

                features[f]->saveFeature(imageList[i],fts);

                totalTime += feature_evaluation_time.stop("");

                fts.release();

                delete im;
            }
        }
    }

    dataset->numFeatureDims = numFeatureDims;
    cout<< "Total Dims of the Feature Vector = \t" << numFeatureDims<<endl;
    feature_evaluation_time.msg(totalTime,     "\t\telapsed time to compute ~ ");
    eval_ftr_time = totalTime / (float)imageList.size();
    feature_evaluation_time.msg(eval_ftr_time, "\t\t              per image ~ ");

}

void Model::RankFeatures(vector<string> &imageList, int from, int to, bool isRank, int &ranking_time){

    if(to== -1) to = imageList.size();
    Mat trainData, sampleData, nonZeroLabels;


    if(dataset->numFeatureDims == 0){

        for(int f=0; f< dataset->totalFeatureType; f++)
            if(features[f]->getFeatureDim() !=0 ){
                dataset->numFeatureDims += features[f]->getFeatureDim();
            }
    }
    if(isRank){

        Timer feature_ordering_time;

        cout<<"Ranking features with mRMR... Total #FeatureSet: "<< dataset->totalFeatureType <<endl;

        for(int i=from; i< to; i++ ){

            vector<Mat> FeatureSet(dataset->totalFeatureType);
            Mat labels;
            Image *groundTruth = new Image(dataset->grFolder+imageList[i]+dataset->grFileExt);
            dataset->RGB2Label(groundTruth->getRGB(),labels);
            delete groundTruth;

            if(sum(labels).val[0] == 0){

                cout<<"\t\t\tWARNING: Unlabeled Image: "<< imageList[i]<< endl;
            }
            else{
                for(int f=0; f< dataset->totalFeatureType; f++ ){

                    Mat ftr;
                    dataset->features[f]->loadFeature(imageList[i],ftr);

                    int subSample = dataset->features[f]->getSubSample();
                    int index = 0; // Get only non zero labeled features

                    for(int r=0; r< labels.rows; r= r+ subSample)
                        for(int c=0; c< labels.cols; c = c+ subSample){

                            if(labels.at<float>(r,c) != 0.0){

                                if(f == 0){
                                    nonZeroLabels.push_back(labels.at<float>(r,c)); // get labels once
                                }
                                FeatureSet[f].push_back(ftr.row(index));
                            }
                            index++;
                        }
                    ftr.release();
                }

                hconcat(FeatureSet,sampleData); // Features for one image
                trainData.push_back(sampleData);
                for(int f=0; f< dataset->totalFeatureType; f++ ) FeatureSet[f].release();
                FeatureSet.clear();
                sampleData.release();
                labels.release();
            }
        }

        // concat labels with feature values
        FILE* file= fopen((dataset->trainFolder+"all_data.csv").c_str(), "w");

        if(file == NULL)    cout<<"ERROR: "<< (dataset->trainFolder+"all_data.csv")<<" not found."<<endl;

        fprintf(file, "class");
        for(int i=0; i< trainData.cols; i++) {fprintf(file,",%s",feature_names_[i].c_str());}
        fprintf(file,"\n");
        for(int l=0; l< nonZeroLabels.rows; l++){
            fprintf(file,"%d",(int)nonZeroLabels.at<float>(l,0));
            for(int t=0; t< trainData.cols; t++){

                fprintf(file,",%f",trainData.at<float>(l,t));
            }
            fprintf(file,"\n");
        }

        fclose(file);

        stringstream nFeatures, datanum;
        nFeatures  << trainData.cols;
        datanum    << trainData.rows + 1;
        string mrmr_command    =    string("./misc/mrmr/mrmr -i "+ dataset->trainFolder+"all_data.csv -n "+nFeatures.str()+" -v "+nFeatures.str()+" -s " + datanum.str() + " -m \"MID\" > "+dataset->trainFolder+dataset->name_+"_mrmr_output.txt");
        cout<< "\trunning mRMR: "<< mrmr_command<<endl;

        trainData.release();
        nonZeroLabels.release();

        feature_ordering_time.start();
        if(system(mrmr_command.c_str())){
            cout<< "Failed to run mrmr..."<<endl;
            exit(EXIT_FAILURE);
        }
        ranking_time = feature_ordering_time.stop("\t\telapsed time to rank ~ ");

    }

    // Read ordered feature indices
    string line;
    ifstream infile;
    infile.open ((dataset->trainFolder+dataset->name_+"_mrmr_output.txt").c_str());
    if(!infile.is_open()){

        cout<<"ERROR: "<< (dataset->trainFolder+dataset->name_+"_mrmr_output.txt")<<" not found."<<endl;
        exit(-1);
    }

    getline(infile,line);
    while(line != "*** mRMR features *** ") getline(infile,line);
    getline(infile,line);

    dataset->selectedFeatures.clear();

    if(dataset->fcount_ == 0 ) dataset->fcount_ = dataset->getFullFeatureDim();

    int n;
    for(int f_ = 0 ; f_ < dataset->fcount_ ; f_++){

        getline(infile, line);
        std::istringstream iss(line);
        iss >> n; // Order
        iss >> n; // Feature Index
        dataset->selectedFeatures.push_back(n-1);

    }

    infile.close();

    //Set selected features for further computation
    int f_dim = 0;
    for(int f=0; f< dataset->totalFeatureType ; f++){

        features[f]->isComputeFeature.setTo(uchar(0));

        for(uint s=0; s< dataset->selectedFeatures.size() ; s++){

            volatile int f_inx = dataset->selectedFeatures[s] - f_dim;

            if((f_inx >= 0 ) && (f_inx < features[f]->getFullDim())){

                features[f]->isComputeFeature.at<char>(0,f_inx) = uchar(1);
//                cout<< features[f]->name_ <<" " << features[f]->isComputeFeature << endl;
            }
        }
        f_dim += features[f]->getFullDim();
    }

    for(int f=0; f< dataset->totalFeatureType ; f++){

        features[f]->setFeatureDim(sum(features[f]->isComputeFeature).val[0]);
    }

    if(isRank)
        cout<< "\tAll Features in Order (N: "<<dataset->fcount_<<"):"<<endl<<"\t | ";
    else
        cout<< "\tSelected Features in Order (N: "<<dataset->fcount_<<"):"<<endl<<"\t | ";
    for(int s=0; s< dataset->fcount_; s++) cout<<feature_names_[dataset->selectedFeatures[s]]<<" | ";
    cout<<endl;
}

void Model::AnalyseFeatures(){

    int from = 0, to =-1;

    if(dataset->numFeatureDims == 0){

        for(int f=0; f< dataset->totalFeatureType; f++)
            if(features[f]->getFeatureDim() !=0 ){
                dataset->numFeatureDims += features[f]->getFeatureDim();
            }
    }

    int howManyFeatures = dataset->numFeatureDims;

    FILE* file = fopen((dataset->resultDir+dataset->name_+"_FeatureAnalysis.txt").c_str(),"w");
    if(file == NULL){

        cout<< "ERROR: Model::AnalyseFeatures : File not opened: "<< (dataset->resultDir+"FeatureAnalysis.txt").c_str() << endl;
        exit(-1);
    }
    fprintf(file,"Index\tFType\tAccuracy\tTraining Time\tEvaluation Time ( Time = HHMMSSmmm )\n");

    int time_classification = -1, time_training = -1;

    if(FAST_COMPUTATION){

        vector<int> subSampleArray(dataset->totalFeatureType);
        cout << "Precomputing Features on the Training Images!" << endl;
        EvaluateFeatures(dataset->trainImgs, from, to);

        cout << "PreComputing Features on the Validation Images!" << endl;
        for(int f=0; f< dataset->totalFeatureType; f++ ){
            subSampleArray[f] = dataset->features[f]->getSubSample();
            dataset->features[f]->setSubSample(0);
        }

//        EvaluateFeatures(dataset->testImgs, from, to);
        EvaluateFeatures(dataset->validationImgs, from, to);

        for(int f=0; f< dataset->totalFeatureType; f++ ){

            dataset->features[f]->setSubSample(subSampleArray[f]);
        }
        subSampleArray.clear();

    }

    for( int f= 1 ; f <= howManyFeatures; f++ ){

        cout<< "*********"<<endl;
        cout<< "FCOUNT: "<< f << endl;
        cout<< "*********"<<endl;

        dataset->fcount_ = f;
        RankFeatures( dataset->trainImgs, from, to, false);
        if(!FAST_COMPUTATION)
            EvaluateFeatures( dataset->trainImgs, from, to);
        TrainPotentials( dataset->trainImgs, from, to, time_training);
        EvaluatePotentials( dataset->validationImgs, from, to, time_classification);
        Confusion( dataset->validationImgs, dataset->detectionFolder, "val_detection_conf_RF.txt", from, to);

        Performance p = getPERFORMANCE();

        fprintf(file,"%d\t%c\t%.3f\t\t%s\t%s\n",f,feature_names_[dataset->selectedFeatures[f-1]].at(0),p.overall ,Timer::formatted_time(time_training).c_str(),Timer::formatted_time(time_classification).c_str());

        // Flush file to read it online on texteditor
        fflush(file);
        fsync(fileno(file));
    }

    fclose(file);
}

int Model::SelectFeatures(float alpha, float beta){

    FILE* file = fopen((dataset->resultDir+dataset->name_+"_FeatureAnalysis.txt").c_str(),"r");
    if(file == NULL){

        cout<< "ERROR: Model::SelectFeatures : File not opened: "<< (dataset->resultDir+dataset->name_+"_FeatureAnalysis.txt").c_str() << endl;
        exit(-1);
    }

    char title[100];
    fgets(title,100,file);

    int fNo, selectedFNo;
    float performance;
    float gain  = 0.0, maxGain = 0.0;

    vector< float > Performances;


    while(fscanf(file,"%d %*c %f %*s %*d\n",&fNo,&performance) != EOF ){

        Performances.push_back(performance);
    }

    fclose(file);

    cout << "n \t (N-n) \t Acc \t=> Gain" << endl;

    for(int i=0; i < Performances.size(); i++){

        gain = pow( Performances[i], alpha ) * pow( fNo - i, 1.0 / beta );
        cout << i+1 << "\t" <<(fNo - i) << "\t" << Performances[i] << "\t=>" << gain;

        if( gain > maxGain ){

            selectedFNo = i+1;
            maxGain = gain;
            cout<< "   ===>SELECTED, MaxGain: "<< maxGain;
        }
        cout<< endl;
    }



    return selectedFNo;
}

/// Activate the loading only relevant features from pre-computed feature files
void Model::ActivateFastComputation(bool isFast){

    this->FAST_COMPUTATION = isFast;
}

void Model::TrainPotentials(vector<string>&imageList, int from, int to, int &train_pt_time){

    if(to== -1) to = imageList.size();

    cout<<"Training potentials..."<<endl;

    for(int p=0; p< dataset->numPotentials; p++ ){

        train_pt_time = potentials[p]->Train(dataset, imageList, from, to, FAST_COMPUTATION);
    }
}

void Model::EvaluatePotentials(vector<string> & imageList, int from, int to, int &eval_pt_time){

    if(to== -1) to = imageList.size();

    cout<<"Evaluating potentials..."<<endl;

    eval_pt_time = 0;
    for(int p=0; p< dataset->numPotentials; p++){

        eval_pt_time += potentials[p]->Evaluate(dataset, imageList, from, to, FAST_COMPUTATION);
    }
}

void Model::Confusion(vector<string> & imageList,string folder, string confusionFileName, int from, int to){
//****** THIS PART HAS BEEN COPIED FROM LADICKY's so that WE CAN COMPARE THE RESULTS ********//
//****** http://www.inf.ethz.ch/personal/ladickyl/ (http://www.inf.ethz.ch/personal/ladickyl/ALE.zip)

    if(to == -1) to= imageList.size();

    Mat grLabels, dtLabels ;
    int *pixTotalClass, *pixOkClass, *confusion, *pixLabel, i ;

    confusion       = new int[dataset->numClasses * dataset->numClasses];
    pixTotalClass   = new int[dataset->numClasses];
    pixOkClass      = new int[dataset->numClasses];
    pixLabel        = new int[dataset->numClasses];

    memset(confusion    , 0, dataset->numClasses * dataset->numClasses * sizeof(int));
    memset(pixTotalClass, 0, dataset->numClasses * sizeof(int));
    memset(pixOkClass   , 0, dataset->numClasses * sizeof(int));
    memset(pixLabel     , 0, dataset->numClasses * sizeof(int));

    int actual_class, predicted_class;
    unsigned int pixTotal = 0, pixOk = 0;

    for(i=from; i< to; i++){

        Image *groundTruth  = new Image( dataset->grFolder  + imageList[i]  + dataset->grFileExt );
        Image *result       = new Image( folder             + imageList[i]  + dataset->grFileExt );

        dataset->RGB2Label( groundTruth->getRGB() ,  grLabels );
        dataset->RGB2Label( result->getRGB()      ,  dtLabels );

        delete groundTruth;
        delete result;


        for(int r=0; r< grLabels.rows; r++){
            for(int c=0; c< grLabels.cols; c++){

                actual_class    =   0;
                predicted_class =   0;
                if ( ( grLabels.at<float>(r,c) != 0.0 ) && ( dtLabels.at<float>(r,c) != 0.0 ) ){

                    actual_class            =   (int)grLabels.at<float>(r,c);
                    predicted_class         =   (int)dtLabels.at<float>(r,c);

                    pixTotal++;
                    pixTotalClass[actual_class - 1]++;
                    pixLabel[predicted_class - 1]++;
                    if(actual_class == predicted_class) pixOk++, pixOkClass[actual_class - 1]++;
                    confusion[(actual_class - 1) * dataset->numClasses + predicted_class - 1]++;
                }
            }
        }
    }

    double average = (double)0.0, waverage = 0.0;

    for(unsigned i = 0; i < dataset->numClasses; i++){

        average += (pixTotalClass[i] == 0) ? 0 : pixOkClass[i] / (double) pixTotalClass[i];
        waverage += (pixTotalClass[i] + pixLabel[i] - pixOkClass[i] == 0) ? 0 : pixOkClass[i] / (double) (pixTotalClass[i] + pixLabel[i] - pixOkClass[i]);
    }

    average /= dataset->numClasses, waverage /= dataset->numClasses;

    _PERFORMANCE.overall    =   (pixTotal != 0) ? pixOk / (double)pixTotal : (double)0.0;
    _PERFORMANCE.average    =   average;
    _PERFORMANCE.waverage   =   waverage;

    if( SAVE_CONFUSION ){

        cout<<"Confusion..."<< dataset->resultDir+confusionFileName<<endl;

        FILE *ff;
        ff = fopen((dataset->resultDir+confusionFileName).c_str(), "w");
        for(unsigned int q = 0 ; q < dataset->numClasses; q++)
        {
            for(unsigned int w = 0; w < dataset->numClasses; w++) fprintf(ff, "%.3f ", (pixTotalClass[q] == 0) ? 0 : (confusion[q * dataset->numClasses + w] / (double)pixTotalClass[q]));
            fprintf(ff, "\n");
        }
        fprintf(ff, "\n");
        for(unsigned int q = 0 ; q < dataset->numClasses; q++) fprintf(ff, "%.3f ", (pixTotalClass[q] + pixLabel[q] - pixOkClass[q] == 0) ? 0 : pixOkClass[q] / (double)(pixTotalClass[q] + pixLabel[q] - pixOkClass[q]));
        fprintf(ff, "\n");
        for(unsigned int q = 0 ; q < dataset->numClasses; q++) fprintf(ff, "%.3f ", (pixTotalClass[q] == 0) ? 0 : pixOkClass[q] / (double)pixTotalClass[q]);
        fprintf(ff, "\n");

        fprintf(ff, "overall %.4f, average %.4f, waverage %.4f\n", (pixTotal != 0) ? pixOk / (double)pixTotal : (double)0.0, average, waverage);
        fclose(ff);

        printf("overall %.3f, average %.3f, waverage %.3f\n", (pixTotal != 0) ? pixOk / (double)pixTotal : (double)0.0, average, waverage);
    }

    delete[] pixTotalClass;
    delete[] pixLabel;
    delete[] pixOkClass;
    delete[] confusion;

}

void Model::Solve(vector<string>& imageList, int from, int to, int &solving_time){

    if(to == -1) to= imageList.size();
    if(SAVE_CONFUSION) cout<<"Solving..."<<endl;
    solving_time = optimizer->Solve(dataset,imageList,from,to, SAVE_CONFUSION);

}

void Model::FindOptimalLambda(vector<string>& imageList,int from, int to){


    if(to == -1) to= imageList.size();

    //*** Find optimum lambda for Variational Optimization

    cout<< "Finding optimal lambda for Variational Optimization"<<endl;
    cout<< "\t\t[lambda\t\toverall\taverage\t time]"<<endl;

    SAVE_CONFUSION = false;
    Performance optimum(0.0);
    Performance previous(0.0);
    Performance current(0.0);
    float optimum_lambda = 0.0;
    float eps_ = 0.005;
    float lambda = 1.0;
    float inc_   = 3;
    bool  in_btw = false;
    int   time_segm = -1;
    do{
        dataset->lambda = lambda;
        this->Solve(imageList, from, to, time_segm);
        this->Confusion(imageList, dataset->resultFolder, "", from, to);
        cout<<setprecision(3)<<"\t\t[  "<<lambda<<setprecision(3)<<"\t\t"<<_PERFORMANCE.overall<<"\t"<<_PERFORMANCE.average<<"\t~"<<setprecision(2)<< time_segm / 1000.0 <<"]"<<endl;
        if(optimum.overall < _PERFORMANCE.overall/* && ((_PERFORMANCE.overall-_PERFORMANCE.average) < 0.05)*/){
            optimum.set(_PERFORMANCE);
            optimum_lambda  = lambda;
        }
        previous.set(current);
        current.set(_PERFORMANCE);
        if((current.overall - previous.overall) > 0){
            if(in_btw){
                inc_ /= 2;
            }
            lambda  += inc_;
            in_btw   = false;
        }
        else{
            inc_ /= 2;
            lambda  -= inc_;
            in_btw   = true;
        }
    }while(std::abs(current.overall-previous.overall) > eps_);

    SAVE_CONFUSION = true;
    dataset->lambda = optimum_lambda;
    cout<<"optimum lambda and performance:\n"<<setprecision(2)<<"\t[  "<<optimum_lambda<<"\t\t"<<optimum.overall<<"\t"<<optimum.average<<"]"<<endl;
    //*** End of optimum lambda

    _PERFORMANCE = optimum;
}
