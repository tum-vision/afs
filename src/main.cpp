
#include <main.hpp>

int main(int argc, char **argv)
{
    /** Set start-end indices**/
    int from = 0, to = -1;

    /** Set the image name for single image evaluation **/
    string name     ="";

    Model *model    = new Model();
    Dataset *dataset= new Test();
    Timer *run_time = new Timer();

    /** Boolean Flags **/
    bool rank       = false;
    bool analyse    = false;
    bool segment    = true;

    string OutputFileName = dataset->resultDir;

    /** Set Model Structure **/
    model->SetStructure(dataset);

    /** If process only one image **/
    if(!name.empty()){

        uint pos =dataset->findImage(name);
        if(pos< dataset->testImgs.size()){
            from= pos;
            to  = pos+1;
        }
    }

    /** Activate Fast Computation **/
    model -> ActivateFastComputation();

    if(rank){

        freopen("/dev/tty","w",stdout);
        redirectCout( model, OutputFileName + "OutputRanking.txt" );

        /** Evaluate and Rank Features **/
        model -> EvaluateFeatures(dataset->trainImgs, from, to);
        model -> RankFeatures(dataset->trainImgs, from, to, true);

    }

    if(analyse){

        freopen("/dev/tty","w",stdout);
        redirectCout( model, OutputFileName + "OutputAnalysis.txt" );

        /** Start Timer **/
        run_time -> start();

        /** Analyse Features **/
        model -> AnalyseFeatures();

        /** Stop Timer **/
        run_time -> stop("Elapsed time to analyse features : ");

    }

    /** Deacticate Fast Computation **/
    model -> ActivateFastComputation(false);

    if(segment){

        freopen("/dev/tty","w",stdout);
        redirectCout( model, OutputFileName + "OutputSegmentation.txt" );

        /** Select Features **/
        float alpha     = 5.0;
        float beta      = 2.0;

        dataset->fcount_= model -> SelectFeatures( alpha, beta );
        cout << "Number of selected features : "<< dataset->fcount_ << endl;

        /** Clean Validation Files **/
        dataset -> clearDir( dataset -> detectionFolder );
        dataset -> clearDir( dataset -> potentialFolder );

        /** Evaluate Segmentations with Selected Features **/

        model -> RankFeatures(dataset->trainImgs, from, to, false);
        model -> EvaluateFeatures(dataset->trainImgs, from, to);
        model -> TrainPotentials(dataset->trainImgs,from,to);
        model -> EvaluatePotentials(dataset->testImgs,from,to);
        model -> Confusion(dataset->testImgs,dataset->detectionFolder,"detection_conf_RF.txt",from,to);
        model -> FindOptimalLambda(dataset->testImgs,from,to);
        model -> Solve(dataset->testImgs,from,to);
        model -> Confusion(dataset->testImgs,dataset->resultFolder,"confusion_RF.txt",from,to);
    }

    if( !rank & !analyse & !segment ){

        /** Select Features **/
        float alpha     = 5.0;
        float beta      = 2.0;

        dataset->fcount_= model -> SelectFeatures( alpha, beta );
        cout << "Number of selected features : "<< dataset->fcount_ << endl;

    }

    delete run_time;
    delete dataset;
    delete model;

    return 0;
}
