#include "colorizerealnbr.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize( eColorizeMode mode, const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    switch ( mode )
    {
        case IronBow: break;
        case RedYellow: break;
        case BlueRed: break;
        case BlackRed: break;
        case BlueRedBin: break;
        case BlueGreenRed: break;
        case Grayscale: break;
        case ShiftGray: break;
    }
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

    for ( unsigned int n = 0; n < dataSize; n++)
    {
            uint16_t data = dataIn[ n ]*1000;

            r = data >> 6;
            if ( r > 255       ) r = 255;
            if ( data & 0x2000 ) g = 0x0ff & ( data >> 5 ); else g = 0;
            b = 0;


            if ( (data & 0x3800) == 0x0000 ) { b = 0x0ff & (( data >> 3 )); }
            if ( (data & 0x3800) == 0x0800 ) { b = 0x0ff & (( data >> 4 )); b = 255 - b; b = b + 127; }
            if ( (data & 0x3800) == 0x1000 ) { b = 0x0ff & (( data >> 4 )); b = 128 - b; }
            if ( (data & 0x3000) == 0x3000 ) { b = 0x0ff & (( data >> 4 )); }

            dataOut[n].r = r;
            dataOut[n].g = g;
            dataOut[n].b = b;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_RedYellow( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r;
    uint16_t g;
    uint16_t b;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ];

        r = data >> 6;
        if (r > 255) r = 255;
        if (data & 0x2000) g = (0x0ff & (data >> 5)); else g = 0;
        b = 0;

        dataOut[n].r = r;
        dataOut[n].g = g;
        dataOut[n].b = b;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlueRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ];

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

        dataOut[n].r = r;
        dataOut[n].g = g;
        dataOut[n].b = b;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlackRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ];

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

        dataOut[n].r = r;
        dataOut[n].g = g;
        dataOut[n].b = b;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlueRedBin( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ];

        r = data >> 6;
        if (r > 255) r = 255;
        b = data >> 6;
        if (b > 255) b = 255;

        dataOut[n].r = r;
        dataOut[n].g = g;
        dataOut[n].b = b;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_BlueGreenRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ];

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

        dataOut[n].r = r;
        dataOut[n].g = g;
        dataOut[n].b = b;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_Grayscale( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        volatile float dataFloat = dataIn[ n ].Distance;
        uint16_t data = dataFloat * 1000;

        r = data >> 6;

        dataOut[n].r = r;
        dataOut[n].g = r;
        dataOut[n].b = r;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRealNumbers::Colorize_ShiftGray( const float* dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift )
{
    uint16_t r = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ];

        r = data >> shift;
        if (r > 255) r = 255;

        dataOut[n].r = r;
        dataOut[n].g = r;
        dataOut[n].b = r;
    }
}
