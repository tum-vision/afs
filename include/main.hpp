#ifndef MAIN_HPP
#define MAIN_HPP

#include <common.hpp>

#include <Model.hpp>
#include <Dataset.hpp>
#include <Image.hpp>
#include <Feature.hpp>
#include <ImageProc.hpp>

/**
 * @brief redirect stdout, cout to the file
 * @param model             pointer to the current Model
 * @param OutputFileName    output file path
 * @return
 */
bool redirectCout(Model *model, string OutputFileName){

    /** Redirect cout to the file **/
    cout << "cout redirected to the file : " << OutputFileName << endl;
    freopen( OutputFileName.c_str(),"w", stdout );
    model -> PrintDatasetInfo();
}


#endif // MAIN_HPP
