#include "colorize_ray_data.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize( const eColorizeMode mode, const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    switch ( mode )
    {
        case IronBow:      Colorize_IronBow(dataIn, dataOut, dataSize); break;
        case RedYellow:    Colorize_RedYellow(dataIn, dataOut, dataSize); break;
        case BlueRed:      Colorize_BlueRed(dataIn, dataOut, dataSize); break;
        case BlackRed:     Colorize_BlackRed(dataIn, dataOut, dataSize); break;
        case BlueRedBin:   Colorize_BlueRedBin(dataIn, dataOut, dataSize); break;
        case BlueGreenRed: Colorize_BlueGreenRed(dataIn, dataOut, dataSize); break;
        case Grayscale:    Colorize_Grayscale(dataIn, dataOut, dataSize); break;
        case ShiftGray:    Colorize_ShiftGray(dataIn, dataOut, dataSize, 0x22); break;
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_IronBow( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
            uint16_t data = dataIn[ n ].Distance * 1000;

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
void ColorizeRayData::Colorize_RedYellow( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r;
    uint16_t g;
    uint16_t b;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ].Distance * 1000;

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
void ColorizeRayData::Colorize_BlueRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ].Distance * 1000;

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
void ColorizeRayData::Colorize_BlackRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ].Distance * 1000;

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
void ColorizeRayData::Colorize_BlueRedBin( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ].Distance * 1000;

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
void ColorizeRayData::Colorize_BlueGreenRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    uint16_t r = 0;
    uint16_t g = 0;
    uint16_t b = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ].Distance * 1000;

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
void ColorizeRayData::Colorize_Grayscale( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
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
void ColorizeRayData::Colorize_ShiftGray( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift )
{
    uint16_t r = 0;

    for ( unsigned int n = 0; n < dataSize; n++)
    {
        uint16_t data = dataIn[ n ].Distance * 1000;

        r = data >> shift;
        if (r > 255) r = 255;

        dataOut[n].r = r;
        dataOut[n].g = r;
        dataOut[n].b = r;
    }
}
