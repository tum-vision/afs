#ifndef IMAGEPROC_HPP
#define IMAGEPROC_HPP

#include <common.hpp>

/**
 * @brief The Image Processing class
 */
class ImageProc{

public:
    /**
     * @brief Class Constructor
     */
    ImageProc();

public:

    /**
     * @brief  compute image edges
     * @param input         RGB image
     * @return edge image
     */
    static Mat  getEdgeImage(Mat input);

    /**
     * @brief  compute integral image
     * @param input             RGB Image
     * @return                  integral image
     */
    static Mat  getIntegralImage(Mat input);

    /**
     * @brief  zero pad to the image
     * @param input             image
     * @param extTopRows        number of top rows to pad zero
     * @param extBotRows        number of bot rows to pad zero
     * @param extLeftCols       number of left cols to pad zero
     * @param extRightCols      number of right cols to pad zero
     * @return                  zero padded image
     */
    static Mat  getPaddedImg(Mat input,int extTopRows,int extBotRows,int extLeftCols, int extRightCols);

    /**
     * @brief  convert image color space
     * @param input         image
     * @param code          OpenCV color conversion code
     * @return              color converted image
     */
    static Mat  convertColor(Mat input,int code);

    /**
     * @brief    get specific channel of image
     * @param input         image
     * @param channel       channel to return
     * @return              image channel
     */
    static Mat  getChannel(Mat input,unsigned char channel);

};

#endif // IMAGEPROC_HPP
