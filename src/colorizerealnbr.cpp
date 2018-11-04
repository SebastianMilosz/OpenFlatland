#include "colorizerealnbr.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize( eColorizeMode mode, const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{

}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_IronBow( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for( unsigned int n = 0; n < dataSize; n++)
    {
/*
            uint16_t data = imgIn.at<uint16_t>( row, col );

            r = data >> 6;
            if( r > 255       ) r = 255;
            if( data & 0x2000 ) g = 0x0ff & ( data >> 5 ); else g = 0;
            b = 0;


            if( (data & 0x3800) == 0x0000 ) { b = 0x0ff & (( data >> 3 )); }
            if( (data & 0x3800) == 0x0800 ) { b = 0x0ff & (( data >> 4 )); b = 255 - b; b = b + 127; }
            if( (data & 0x3800) == 0x1000 ) { b = 0x0ff & (( data >> 4 )); b = 128 - b; }
            if( (data & 0x3000) == 0x3000 ) { b = 0x0ff & (( data >> 4 )); }

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = g;
            imgOut.at<cv::Vec3b>( row, col )[2] = b;
*/
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_RedYellow( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_RedYellow empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r;
    uint16_t g;
    uint16_t b;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            r = data >> 6;
            if (r > 255) r = 255;
            if (data & 0x2000) g = (0x0ff & (data >> 5)); else g = 0;
            b = 0;

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = g;
            imgOut.at<cv::Vec3b>( row, col )[2] = b;
        }
    }

    SetCVMAT( imgOut );
*/
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlueRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_BlueRed empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            if( (data & 0x3000) == 0 )
            {
                g = 0;
                b = 0x07f & (data >> 5);
                b = b + 128;
                r = 0;
            }
            else if( (data & 0x3000) == 0x1000 )
            {
                b = 0x07f & (data >> 5);
                b = 255 - b;
                g = 0;
                r = 0x07f & (data >> 5);
            }
            else if( (data & 0x3000) == 0x2000 )
            {
                b = 0x07f & (data >> 5);
                b = 128 - b;
                r = 0x07f & (data >> 5);
                r = r + 128;
                g = 0;
            }
            else if((data & 0x3000) == 0x3000)
            {
                b = 0;
                r = 255;
                g = 0x0ff & (data >> 4);
            }

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = g;
            imgOut.at<cv::Vec3b>( row, col )[2] = b;
        }
    }

    SetCVMAT( imgOut );
*/
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlackRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_BlackRed empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            if ((data & 0x3000) == 0)
            {
                g = 0;
                b = (data >> 4);
                b = b & 0xff;
                r = 0;
            }
            else if ((data & 0x3000) == 0x1000)
            {
                b = 0x07f & (data >> 5);
                b = 255 - b;
                g = 0;
                r = 0x07f & (data >> 5);
            }
            else if ((data & 0x3000) == 0x2000)
            {
                b = 0x07f & (data >> 5);
                b = 128 - b;
                r = 0x07f & (data >> 5);
                r = r + 128;
                g = 0;
            }
            else if ((data & 0x3000) == 0x3000)
            {

                b = 0;
                r = 255;
                g = 0x0ff & (data >> 4);
            }

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = g;
            imgOut.at<cv::Vec3b>( row, col )[2] = b;
        }
    }

    SetCVMAT( imgOut );
*/
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlueRedBin( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_BlueRedBin empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            r = data >> 6;
            if (r > 255) r = 255;
            b = data >> 6;
            if (b > 255) b = 255;

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = g;
            imgOut.at<cv::Vec3b>( row, col )[2] = b;
        }
    }

    SetCVMAT( imgOut );
*/
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlueGreenRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_BlueGreenRed empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            if ((data & 0x3000) == 0)
            {
                b = 255;
                g = 0x0ff & (data >> 4);
                r = 0;
            }
            else if ((data & 0x3000) == 0x1000)
            {
                b = 0x0ff & (data >> 4);
                b = 255 - b;
                g = 255;
                r = 0;
            }
            else if ((data & 0x3000) == 0x2000)
            {
                b = 0;
                g = 255;
                r = 0x0ff & (data >> 4);
            }
            else if ((data & 0x3000) == 0x3000)
            {

                b = 0;
                g = 0x0ff & (data >> 4);
                g = 255 - g;
                r = 255;
            }

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = g;
            imgOut.at<cv::Vec3b>( row, col )[2] = b;
        }
    }

    SetCVMAT( imgOut );
*/
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_Grayscale( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_Grayscale empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            r = data >> 6;

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = r;
            imgOut.at<cv::Vec3b>( row, col )[2] = r;
        }
    }

    SetCVMAT( imgOut );
*/
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_ShiftGray( const float* dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift )
{
/*
    cMutexLocker locker( &m_mutex );

    cv::Mat& imgIn  = GetCVMAT();
    cv::Mat  imgOut;

    if( imgIn.empty() ) { LogSingleShotError( 0x04, "Colorize_ShiftGray empty()" ); return; }

    imgOut.create(imgIn.rows, imgIn.cols, CV_8UC3);

    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for(int row = 0; row < imgIn.rows; row++)
    {
        for(int col = 0; col < imgIn.cols; col++)
        {
            uint16_t data = imgIn.at<uint16_t>( row, col );

            r = data >> shift;
            if (r > 255) r = 255;

            imgOut.at<cv::Vec3b>( row, col )[0] = r;
            imgOut.at<cv::Vec3b>( row, col )[1] = r;
            imgOut.at<cv::Vec3b>( row, col )[2] = r;
        }
    }

    SetCVMAT( imgOut );
*/
}
