
#include <Dataset.hpp>

Dataset::Dataset(){

    isSplitDataset    = true;
    showEntropy       = false;
    saveDetections    = true;
    savePotentials    = false;
    index             = 0;
    optimizer         = new VariationalOptimization();

}

Dataset::~Dataset(){

     allImages.clear();
     trainImgs.clear();
     testImgs.clear();
}


void Dataset::createResultFolders(){

    // RESULT FOLDERS

    resultDir       = mainFolder+"Result/";

    trainFolder     = resultDir+"Train/";
    resultFolder    = resultDir+"Segmentation/";

    haarFolder      = resultDir+"Feature/Haar/";
    haarFeatureExt  = ".haar";

    colorFolder    = resultDir+"Feature/Color/";
    colorFeatureExt= ".color";

    locationFolder       = resultDir+"Feature/Location/";
    locationFeatureExt   = ".loc";

    textonFolder       = resultDir+"Feature/Texton/";
    textonFeatureExt   = ".tex";

    potentialFolder = resultDir+"DenseXX/";
    detectionFolder = resultDir+"Detections/";
    potentialExt    = ".dns";

    pMapFolder      = resultDir+"PotentialMaps/";

    createDir(resultDir);
    createDir(trainFolder);
    createDir(resultFolder);

    createDir(haarFolder);
    createDir(colorFolder);
    createDir(locationFolder);
    createDir(textonFolder);

    createDir(potentialFolder);
    createDir(detectionFolder);

    if(savePotentials) createDir(pMapFolder);

}

int Dataset::getFullFeatureDim(){

    int num = 0;
    for(int i =0; i < totalFeatureType; i++){

        num += features[i]->getFullDim();
    }
    return num;
}

void Dataset::removeUnlabeledImages(){

    for(uint i = 0; i < allImages.size() ; i ++){

        Mat labels;
        Image *groundTruth = new Image( grFolder + allImages[i] + grFileExt );
        RGB2Label(groundTruth->getRGB(), labels );
        delete groundTruth;
        if(sum(labels).val[0] == 0){

            cout<<remove(path(imageFolder + allImages[i] + imageFileExt));
            cout<<remove(path(grFolder + allImages[i] + grFileExt));
            cout<<remove(path(depthFolder + allImages[i] + depthFileExt));
            cout<<"\t\t\tWARNING: NONLABELED IMAGE: "<< allImages[i]<<endl;
        }
    }
}

void Dataset::RGB2Label(Mat rgb, Mat& labels){

    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 0;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            for(int k = 0; k < 8; k++)
                l  = (l << 3) | (((r >> k) & 1) << 0) | (((g >> k) & 1) << 1) | (((b >> k) & 1) << 2);

            labels.at<float>(i,j) = l;
        }
}
void Dataset::Label2RGB(Mat labels, Mat & rgb){

    rgb= Mat(labels.rows,labels.cols,CV_8UC3);

    for(int i=0; i< labels.rows; i++)
        for(int j=0; j< labels.cols; j++)
        {
            unsigned char l = labels.at<unsigned char>(i,j);
            unsigned char r,g,b;

            r=0;g=0;b=0;

            for(int k = 0; l > 0; k++, l >>= 3){

                r |= (unsigned char) (((l >> 0) & 1) << (7 - k));
                g |= (unsigned char) (((l >> 1) & 1) << (7 - k));
                b |= (unsigned char) (((l >> 2) & 1) << (7 - k));
            }
            rgb.at<Vec3b>(i,j)[2] = r;
            rgb.at<Vec3b>(i,j)[1] = g;
            rgb.at<Vec3b>(i,j)[0] = b;
        }
}
void Dataset::getDatasetImages(int trainCount, int testCount){

    vector<string>::const_iterator first;
    vector<string>::const_iterator last;

    if(!exists(this->imageFolder)){
        cout << "Image Folder <"<<this->imageFolder<<"> not found !\n";
        exit(-1);
    }

    this->trainCount = trainCount;
    this->testCount  = testCount;

    if(isSplitDataset){

        if(!readSplitsToVectors() || ( ( testCount !=0 ) && ( (uint)testCount != testImgs.size() ) ) ) {

            allImages.clear();
            readImgsInDir(allImages,imageFolder,imageFileExt);

            srand ( unsigned ( time(0) ) );
            random_shuffle( allImages.begin() , allImages.end() );

            int totalCount = allImages.size();

            if(trainCount == 0 || testCount ==0){
                if(totalCount % 2 == 0)
                {
                    trainImgs = vector<string> ( allImages.begin(), allImages.end() - round(totalCount / 2) );
                    testImgs  = vector<string> ( allImages.begin() + round(totalCount / 2), allImages.end() );
                }
                else
                {
                    trainImgs = vector<string> ( allImages.begin(), allImages.end() - round(totalCount / 2)     );
                    testImgs  = vector<string> ( allImages.begin() + round(totalCount / 2) + 1, allImages.end() );
                }

            }
            else{
                trainImgs = vector<string> (  allImages.begin(), allImages.end() - testCount/*round(totalCount / 2)*/);
                testImgs  = vector<string> ( allImages.begin() + trainCount/*round(totalCount / 2)*/,allImages.end() );
            }

            first = trainImgs.end() - trainImgs.size() * 0.2;
            last  = trainImgs.end();
            validationImgs = vector<string>(first, last);
            trainImgs.erase(trainImgs.end() - trainImgs.size() * 0.2, trainImgs.end());

            writeSplitsToFile();
        }
    }
    else
    {
        readImgsInDir(trainImgs,imageFolder,imageFileExt);
        readImgsInDir(testImgs,imageFolder,imageFileExt);

        allImages.reserve(trainImgs.size() + testImgs.size());
        allImages.insert(allImages.end(),trainImgs.begin(),trainImgs.end());
        allImages.insert(allImages.end(),testImgs.begin(),testImgs.end());
    }

}

void Dataset::readImgsInDir(vector<string>& imgs, string folder, string ext){

    if (!exists(folder)) return;

    if (is_directory(folder))
    {
      recursive_directory_iterator it(folder);
      recursive_directory_iterator endit;
      while(it != endit)
      {
        if (is_regular_file(*it) and it->path().extension() == ext)
        {
            imgs.push_back(it->path().stem().string());
        }
        ++it;
      }
    }
}

bool Dataset::readSplitsToVectors(){

    if(exists(path(mainFolder+"trainList.txt"))){

        std::ifstream input_train((mainFolder+"trainList.txt").c_str());
        std::copy(std::istream_iterator<std::string>(input_train),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(trainImgs));

        std::ifstream input_validation((mainFolder+"validationList.txt").c_str());
        std::copy(std::istream_iterator<std::string>(input_validation),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(validationImgs));

        std::ifstream input_test((mainFolder+"testList.txt").c_str());
        std::copy(std::istream_iterator<std::string>(input_test),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(testImgs));

        allImages.reserve(trainImgs.size() + validationImgs.size() + testImgs.size());
        allImages.insert(allImages.end(),trainImgs.begin(),trainImgs.end());
        allImages.insert(allImages.end(),validationImgs.begin(),validationImgs.end());
        allImages.insert(allImages.end(),testImgs.begin(),testImgs.end());

        input_train.close();
        input_validation.close();
        input_test.close();

        return true;
    }

    return false;
}

void Dataset::writeSplitsToFile(){

    std::ofstream output_train((mainFolder+"trainList.txt").c_str());
    std::ostream_iterator<std::string> output_iterator_tr(output_train, "\n");
    std::copy(trainImgs.begin(), trainImgs.end(), output_iterator_tr);

    std::ofstream output_validation((mainFolder+"validationList.txt").c_str());
    std::ostream_iterator<std::string> output_iterator_vl(output_validation, "\n");
    std::copy(validationImgs.begin(), validationImgs.end(), output_iterator_vl);

    std::ofstream output_test((mainFolder+"testList.txt").c_str());
    std::ostream_iterator<std::string> output_iterator_test(output_test, "\n");
    std::copy(testImgs.begin(), testImgs.end(), output_iterator_test);

    output_train.close();
    output_validation.close();
    output_test.close();
}

int Dataset::clearDir(string dirName){

    uintmax_t ok = 0;
    path path_to_remove(dirName);
    for (directory_iterator end_dir_it, it(path_to_remove); it!=end_dir_it; ++it) {
        ok &= remove_all(it->path());
    }
    return ok;
}
int Dataset::createDir(string dirName){

    if(!exists(dirName))   return create_directories(dirName);
    else                   return 1;
}
int  Dataset::findImage(string name){

    return find(testImgs.begin(), testImgs.end(), name) - testImgs.begin();
}
eTrims::eTrims() : Dataset(){

    name_                       = "eTrims";
    mainFolder                  = "../Dataset/eTrims/";
    imageFolder                 = mainFolder+"Images/";
    imageFileExt                = ".jpg";
    grFolder                    = mainFolder+"GroundTruth/";
    grFileExt                   = ".png";

    getDatasetImages(40,20);
    createResultFolders();


    numClasses                  = 8;
    class_weights               = 0;

    // FEATURES

    subSample                   = 5;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(25,25));
    haarFeatures                = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features

    vector<Size> colorPatches;
    colorPatches.push_back(Size(25,25));
    colorFeatures               = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures            = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth             = 1.0;
    textonFeatures              = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    index                       = 0;
    totalFeatureType            = 4;
    features                    = new Feature *[totalFeatureType];
    features[index++]           = haarFeatures;
    features[index++]           = colorFeatures;
    features[index++]           = locationFeatures;
    features[index++]           = textonFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];

    index           = 0;

    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 15;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 50;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n               = numClasses;
    maxSteps        = 3000;
    lambda          = 3.0;
}

void eTrims::RGB2Label(Mat rgb, Mat& labels)
{
    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 0;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            for(int k = 0; k < 8; k++)
                l  = (l << 3) | (((r >> k) & 1) << 0) | (((g >> k) & 1) << 1) | (((b >> k) & 1) << 2);

            if(l == 17) l = 8;
            labels.at<float>(i,j) = l;
        }
}


void eTrims::Label2RGB(Mat labels, Mat& rgb)
{

    rgb= Mat(labels.rows,labels.cols,CV_8UC3);

    for(int i=0; i< labels.rows; i++)
        for(int j=0; j< labels.cols; j++)
        {
            unsigned char l = labels.at<unsigned char>(i,j);
            unsigned char r,g,b;

            r=0;g=0;b=0;

            if(l == 8) l =17;

            for(int k = 0; l > 0; k++, l >>= 3){

                r |= (unsigned char) (((l >> 0) & 1) << (7 - k));
                g |= (unsigned char) (((l >> 1) & 1) << (7 - k));
                b |= (unsigned char) (((l >> 2) & 1) << (7 - k));
            }
            rgb.at<Vec3b>(i,j)[2] = r;
            rgb.at<Vec3b>(i,j)[1] = g;
            rgb.at<Vec3b>(i,j)[0] = b;
        }
}

NYUv1::NYUv1() : Dataset(){

    name_                       = "NYUv1";
    mainFolder                  = "/work/Dataset/NYUv1/";
    imageFolder                 = mainFolder+"Images/";
    imageFileExt                = ".png";
    grFolder                    = mainFolder+"GroundTruth/";
    grFileExt                   = ".png";

    depthFolder                 = mainFolder+"Depth/";
    depthFileExt                = ".png";

    getDatasetImages( 0 , 0 ); // 50%-50%
    createResultFolders();

    // For Depth Feature
    depthFeatureFolder          = resultDir+"Feature/Depth/";
    depthFeatureExt             = ".depth";
    createDir(depthFeatureFolder);

    numClasses                  = 12;
    class_weights               = 0;

    // FEATURES

    subSample                   = 5;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(25,25));
    haarFeatures                = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(25,25));
    colorFeatures               = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures            = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth             = 1.0;
    textonFeatures              = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    // Depth Features
    vector<Size> depthPatches;
    depthPatches.push_back(Size(25,25));
    depthFeatures               = new DepthFeatures(depthPatches, depthFeatureFolder, depthFeatureExt, subSample);

    index                       = 0;
    totalFeatureType            = 5;
    features                    = new Feature *[totalFeatureType];
    features[index++]           = haarFeatures;
    features[index++]           = colorFeatures;
    features[index++]           = locationFeatures;
    features[index++]           = textonFeatures;
    features[index++]           = depthFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];

    index       = 0;

    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 10;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 8;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n           = numClasses;
    maxSteps    = 3000;
    lambda      = 3.0;
}

void NYUv1::RGB2Label(Mat rgb, Mat &labels){

    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 0;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            for(int k = 0; k < 8; k++)
                l  = (l << 3) | (((r >> k) & 1) << 0) | (((g >> k) & 1) << 1) | (((b >> k) & 1) << 2);

            if     (l == 255 ) l = 0;
            else if(l >= 8   ) l -=2;
            else if(l >= 5   ) l -=1;
            labels.at<float>(i,j) = l;
        }
}
void NYUv1::Label2RGB(Mat labels, Mat &rgb){

    rgb= Mat(labels.rows,labels.cols,CV_8UC3);

    for(int i=0; i< labels.rows; i++)
        for(int j=0; j< labels.cols; j++)
        {
            unsigned char l = labels.at<unsigned char>(i,j);
            unsigned char r,g,b;

            r=0;g=0;b=0;

            if      ( l >  6 ) l += 2;
            else if ( l >  4 ) l += 1;

            for(int k = 0; l > 0; k++, l >>= 3){

                r |= (unsigned char) (((l >> 0) & 1) << (7 - k));
                g |= (unsigned char) (((l >> 1) & 1) << (7 - k));
                b |= (unsigned char) (((l >> 2) & 1) << (7 - k));
            }

            rgb.at<Vec3b>(i,j)[2] = r;
            rgb.at<Vec3b>(i,j)[1] = g;
            rgb.at<Vec3b>(i,j)[0] = b;

        }  
}

NYUv2::NYUv2() : Dataset(){

    name_                   = "NYUv2";
    mainFolder              = "/work/Dataset/NYUv2/";
    imageFolder             = mainFolder+"Images/";
    imageFileExt            = ".png";
    grFolder                = mainFolder+"GroundTruth/";
    grFileExt               = ".png";

    depthFolder             = mainFolder+"Depth/";
    depthFileExt            = ".png";

    getDatasetImages( 795 , 654 );
    createResultFolders();

    // For Depth Feature

    depthFeatureFolder      = resultDir+"Feature/Depth/";
    depthFeatureExt         = ".depth";
    createDir(depthFeatureFolder);

    numClasses              = 13;
    class_weights           = 0;

    // FEATURES

    subSample               = 5;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(25,25));
    haarFeatures            = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(25,25));
    colorFeatures           = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures        = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth         = 1.0;
    textonFeatures          = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    // Depth Features
    vector<Size> depthPatches;
    depthPatches.push_back(Size(25,25));
    depthFeatures           = new DepthFeatures(depthPatches, depthFeatureFolder, depthFeatureExt, subSample);

    index                   = 0;
    totalFeatureType        = 5;
    features                = new Feature *[totalFeatureType];
    features[index++]       = haarFeatures;
    features[index++]       = colorFeatures;
    features[index++]       = locationFeatures;
    features[index++]       = textonFeatures;
    features[index++]       = depthFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];

    index           = 0;

    // Dense Unary Pixel Potential

    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 10;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 8;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n               = numClasses;
    maxSteps        = 3000;
    lambda          = 3.0;
}

void NYUv2::RGB2Label(Mat rgb, Mat &labels){

    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 0;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            for(int k = 0; k < 8; k++)
                l  = (l << 3) | (((r >> k) & 1) << 0) | (((g >> k) & 1) << 1) | (((b >> k) & 1) << 2);

            if     (l == 255 ) l = 0;
            else if(l >= 8   ) l -=2;
            else if(l >= 5   ) l -=1;
            labels.at<float>(i,j) = l;
        }
}
void NYUv2::Label2RGB(Mat labels, Mat &rgb){

        rgb= Mat(labels.rows,labels.cols,CV_8UC3);

        for(int i=0; i< labels.rows; i++)
            for(int j=0; j< labels.cols; j++)
            {
                unsigned char l = labels.at<unsigned char>(i,j);
                unsigned char r,g,b;

                r=0;g=0;b=0;

                if      ( l >  6 ) l += 2;
                else if ( l >  4 ) l += 1;

                for(int k = 0; l > 0; k++, l >>= 3){

                    r |= (unsigned char) (((l >> 0) & 1) << (7 - k));
                    g |= (unsigned char) (((l >> 1) & 1) << (7 - k));
                    b |= (unsigned char) (((l >> 2) & 1) << (7 - k));
                }

                rgb.at<Vec3b>(i,j)[2] = r;
                rgb.at<Vec3b>(i,j)[1] = g;
                rgb.at<Vec3b>(i,j)[0] = b;

            }
    }
MSRC::MSRC() : Dataset(){

    name_                       = "MSRC";
    mainFolder                  = "../Dataset/MSRC/";
    imageFolder                 = mainFolder+"Images/";
    imageFileExt                = ".bmp";
    grFolder                    = mainFolder+"GroundTruth/";
    grFileExt                   = ".bmp";

    getDatasetImages(0,0);
    createResultFolders();

    numClasses                  = 21;
    class_weights               = 0;

    // FEATURES

    subSample                   = 3;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(11,11));
    haarFeatures                = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(11,11));
    colorFeatures               = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures            = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth             = 1.0;
    textonFeatures              = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    index                       = 0;
    totalFeatureType            = 4;
    features                    = new Feature *[totalFeatureType];
    features[index++]           = haarFeatures;
    features[index++]           = colorFeatures;
    features[index++]           = locationFeatures;
    features[index++]           = textonFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];

    index           = 0;


    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 15;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 50;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n               = numClasses;
    maxSteps        = 3000;
    lambda          = 4;
}

void MSRC::RGB2Label(Mat rgb, Mat& labels)
{
    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 0;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            for(int k = 0; k < 8; k++)
                l  = (l << 3) | (((r >> k) & 1) << 0) | (((g >> k) & 1) << 1) | (((b >> k) & 1) << 2);

            if((l == 5) || (l == 8) || (l == 19) || (l == 20)) l = 0;
            if(l > 20) l -= 4;
            else if(l > 8) l -= 2;
            else if(l > 5) l--;
            labels.at<float>(i,j) = l;
        }
}


void MSRC::Label2RGB(Mat labels, Mat& rgb)
{

    rgb= Mat(labels.rows,labels.cols,CV_8UC3);

    for(int i=0; i< labels.rows; i++)
        for(int j=0; j< labels.cols; j++)
        {
            unsigned char l = labels.at<unsigned char>(i,j);
            unsigned char r,g,b;

            r=0;g=0;b=0;

            if(l > 16) l += 4;
            else if(l > 6) l += 2;
            else if(l > 4) l++;

            for(int k = 0; l > 0; k++, l >>= 3){

                r |= (unsigned char) (((l >> 0) & 1) << (7 - k));
                g |= (unsigned char) (((l >> 1) & 1) << (7 - k));
                b |= (unsigned char) (((l >> 2) & 1) << (7 - k));
            }
            rgb.at<Vec3b>(i,j)[2] = r;
            rgb.at<Vec3b>(i,j)[1] = g;
            rgb.at<Vec3b>(i,j)[0] = b;
        }
}

Corel::Corel() : Dataset(){

    name_                       = "Corel";
    mainFolder                  = "../Dataset/Corel/";
    imageFolder                 = mainFolder+"Images/";
    imageFileExt                = ".bmp";
    grFolder                    = mainFolder+"GroundTruth/";
    grFileExt                   = ".bmp";

    getDatasetImages( 60 , 40 );
    createResultFolders();

    numClasses                  = 7;
    class_weights               = 0;

    // FEATURES

    subSample                   = 3;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(11,11));
    haarFeatures                = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(11,11));
    colorFeatures               = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures            = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth             = 1.0;
    textonFeatures              = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    index                       = 0;
    totalFeatureType            = 4;
    features                    = new Feature *[totalFeatureType];
    features[index++]           = haarFeatures;
    features[index++]           = colorFeatures;
    features[index++]           = locationFeatures;
    features[index++]           = textonFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];

    index           = 0;

    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 15;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 50;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n               = numClasses;
    maxSteps        = 3000;
    lambda          = 7.0;
}

void Corel::RGB2Label(Mat rgb, Mat& labels)
{
    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 0;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            for(int k = 0; k < 8; k++)
                l  = (l << 3) | (((r >> k) & 1) << 0) | (((g >> k) & 1) << 1) | (((b >> k) & 1) << 2);

            labels.at<float>(i,j) = l;
        }
}


void Corel::Label2RGB(Mat labels, Mat& rgb)
{

    rgb= Mat(labels.rows,labels.cols,CV_8UC3);

    for(int i=0; i< labels.rows; i++)
        for(int j=0; j< labels.cols; j++)
        {
            unsigned char l = labels.at<unsigned char>(i,j);
            unsigned char r,g,b;

            r=0;g=0;b=0;

            for(int k = 0; l > 0; k++, l >>= 3){

                r |= (unsigned char) (((l >> 0) & 1) << (7 - k));
                g |= (unsigned char) (((l >> 1) & 1) << (7 - k));
                b |= (unsigned char) (((l >> 2) & 1) << (7 - k));
            }

            rgb.at<Vec3b>(i,j)[2] = r;
            rgb.at<Vec3b>(i,j)[1] = g;
            rgb.at<Vec3b>(i,j)[0] = b;
        }
}

Sowerby::Sowerby() : Dataset(){

    name_                       = "Sowerby";
    mainFolder                  = "../Dataset/Sowerby/";
    imageFolder                 = mainFolder+"Images/";
    imageFileExt                = ".png";
    grFolder                    = mainFolder+"GroundTruth/";
    grFileExt                   = ".png";

    getDatasetImages( 60, 44 );
    createResultFolders();

    numClasses                  = 7;
    class_weights               = 0;

    // FEATURES

    subSample                   = 3;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(7,7));
    haarFeatures                = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(7,7));
    colorFeatures              = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures            = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth             = 1.0;
    textonFeatures              = new TextonFeatures(textonFolder,textonFeatureExt,subSample,textonBandWidth);

    index                       = 0;
    totalFeatureType            = 4;
    features                    = new Feature *[totalFeatureType];
    features[index++]           = haarFeatures;
    features[index++]           = colorFeatures;
    features[index++]           = locationFeatures;
    features[index++]           = textonFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];
    index           = 0;

    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 15;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 50;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n               = numClasses;
    maxSteps        = 3000;
    lambda          = 1.0;
}

PedestrianParsing::PedestrianParsing() : Dataset(){

    name_                   = "PedestrianParsing";
    mainFolder              = "../Dataset/PedestrianParsing/";
    imageFolder             = mainFolder+"Images/";
    imageFileExt            = ".png";
    grFolder                = mainFolder+"GroundTruth/";
    grFileExt               = ".png";

    getDatasetImages(101,68);
    createResultFolders();

    numClasses              = 8;
    class_weights           = 0;

    // FEATURES

    subSample               = 5;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(11,11));
    haarFeatures            = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(11,11));
    colorFeatures           = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures        = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth         = 1.0;
    textonFeatures          = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    index                   = 0;
    totalFeatureType        = 4;
    features                = new Feature *[totalFeatureType];
    features[index++]       = haarFeatures;
    features[index++]       = colorFeatures;
    features[index++]       = locationFeatures;
    features[index++]       = textonFeatures;

    // POTENTIALS

    numPotentials      = 1;
    potentials          = new Potential *[numPotentials];

    index               = 0;

    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 15;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 50;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n                   = numClasses;
    maxSteps            = 3000;
    lambda              = 7.0;
}

void PedestrianParsing::RGB2Label(Mat rgb, Mat& labels)
{
    labels= Mat(rgb.rows,rgb.cols,CV_32F,Scalar(0,0,0));

    for(int i=0; i< rgb.rows; i++)
        for(int j=0; j< rgb.cols; j++)
        {
            unsigned char l = 1;
            unsigned char r = rgb.at<Vec3b>(i,j)[0];
            unsigned char g = rgb.at<Vec3b>(i,j)[1];
            unsigned char b = rgb.at<Vec3b>(i,j)[2];

            switch(r){

                case 0: {
                            if (g == (unsigned char)(45))
                                l = 2 ;

                            else if (g == (unsigned char)(195))
                                l = 3 ;

                        }     break;

                case 90 : l = 4 ; break;
                case 240: l = 5 ; break;
                case 255: l = 6 ; break;
                case 210: l = 7 ; break;
                case 180: l = 8 ; break;

            }

            labels.at<float>(i,j) = l ;
        }
}


void PedestrianParsing::Label2RGB(Mat labels, Mat& rgb)
{

    rgb= Mat(labels.rows,labels.cols,CV_8UC3);

    for(int i=0; i< labels.rows; i++)
        for(int j=0; j< labels.cols; j++)
        {
            unsigned char l = labels.at<unsigned char>(i,j);
            unsigned char r,g,b;

            r=0;g=0;b=0;

            switch (l){

                case 1: r = (unsigned char)(0)  ; g = (unsigned char)(0)  ; b = (unsigned char)(150); break;
                case 2: r = (unsigned char)(0)  ; g = (unsigned char)(45) ; b = (unsigned char)(255); break;
                case 3: r = (unsigned char)(0)  ; g = (unsigned char)(195); b = (unsigned char)(255); break;
                case 4: r = (unsigned char)(90) ; g = (unsigned char)(255); b = (unsigned char)(165); break;
                case 5: r = (unsigned char)(240); g = (unsigned char)(255); b = (unsigned char)(15) ; break;
                case 6: r = (unsigned char)(255); g = (unsigned char)(105); b = (unsigned char)(0)  ; break;
                case 7: r = (unsigned char)(210); g = (unsigned char)(0)  ; b = (unsigned char)(0)  ; break;
                case 8: r = (unsigned char)(180); g = (unsigned char)(0)  ; b = (unsigned char)(0)  ; break;

            }

            rgb.at<Vec3b>(i,j)[0] = b;
            rgb.at<Vec3b>(i,j)[1] = g;
            rgb.at<Vec3b>(i,j)[2] = r;
        }
}

Test::Test() : Dataset(){

    name_                   = "Test";
    mainFolder              = "../Dataset/Test/";
    imageFolder             = mainFolder+"Images/";
    imageFileExt            = ".png";
    grFolder                = mainFolder+"GroundTruth/";
    grFileExt               = ".png";

    getDatasetImages( 60, 44 );
    createResultFolders();

    numClasses              = 7;
    class_weights           = 0;

    // FEATURES

    subSample               = 3;
    // Haar Features
    vector<Size> haarPatches;
    haarPatches.push_back(Size(7,7));
    haarFeatures            = new HaarLikeFeatures(haarPatches,haarFolder,haarFeatureExt,subSample);

    // Color Features
    vector<Size> colorPatches;
    colorPatches.push_back(Size(7,7));
    colorFeatures           = new ColorFeatures(colorPatches,colorFolder,colorFeatureExt,subSample);

    // Location Features
    locationFeatures        = new LocationFeatures(locationFolder,locationFeatureExt,subSample);

    // Texton Features
    textonBandWidth         = 1.0;
    textonFeatures          = new TextonFeatures(textonFolder,textonFeatureExt,subSample, textonBandWidth);

    index                   = 0;
    totalFeatureType        = 4;
    features                = new Feature *[totalFeatureType];
    features[index++]       = haarFeatures;
    features[index++]       = colorFeatures;
    features[index++]       = locationFeatures;
    features[index++]       = textonFeatures;

    // POTENTIALS

    numPotentials   = 1;
    potentials      = new Potential *[numPotentials];
    index           = 0;

    // Dense Unary Pixel Potential
    DenseUnaryPixelPotential* pixelPotential    = new DenseUnaryPixelPotential(potentialFolder,potentialExt);
    potentials[index]                           = pixelPotential;

    // Random Forest Parameters
    RFParams.max_depth                          = 15;
    RFParams.min_sample_count                   = 10;
    RFParams.max_categories                     = 15;
    RFParams.use_surrogates                     = false;
    RFParams.regression_accuracy                = 0;
    RFParams.calc_var_importance                = false;
    RFParams.nactive_vars                       = 0;
    RFforest_accuracy                           = 0.01;
    RFmax_num_of_trees_in_the_forest            = 50;
    RFParams.term_crit                          = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, RFmax_num_of_trees_in_the_forest, RFforest_accuracy);
    RFParams.priors                             = class_weights;

    potentials[index]->learning                 = new RandomForest(trainFolder+"randomforest.cls", RFParams , numClasses);

    // OPTIMIZATION Parameters

    n               = numClasses;
    maxSteps        = 3000;
    lambda          = 1.0;
}
