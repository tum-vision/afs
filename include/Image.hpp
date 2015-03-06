#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <common.hpp>
#include <ImageProc.hpp>

/**
 * @brief The Image class
 */
class Image{

public:

    /**
     * @brief Class Constructor
     * @param impath    Path to the image
     * @param depthpath Path to the depth image
     */
    Image(string impath, string depthpath = "");

    /**
     * @brief Class DeConstructor
     */
    ~Image();

    /**
     * @brief get image width
     * @return          image width
     */
    int getWidth();

    /**
     * @brief get image height
     * @return          image height
     */
    int getHeight();

    /**
     * @brief get image
     * @return          image
     */
    Mat getImage();

    /**
     * @brief get RGB image
     * @return          RGB image
     */
    Mat getRGB();

    /**
     * @brief is image empty ?
     * @return          true or false
     */
    bool isEmpty();

    /**
     * @brief image path
     */
    string imName;

public:

    /**
     * @brief BGR image
     */
    Mat BGRImage_;

    /**
     * @brief Lab image
     */
    Mat LABImage_;

    /**
     * @brief L channel
     */
    Mat L;

    /**
     * @brief L channel, unsigned 8 bit
     */
    Mat L8U;

    /**
     * @brief a channel
     */
    Mat a;

    /**
     * @brief b channel
     */
    Mat b;

    /**
     * @brief depth image
     */
    Mat depth;
};

#endif // IMAGE_HPP
