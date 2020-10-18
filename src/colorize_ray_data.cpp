#include "colorize_ray_data.hpp"
#include "colorize_number.hpp"

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
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_IronBown<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_RedYellow( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_RedYellow<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_BlueRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_BlueRed<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_BlackRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_BlackRed<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_BlueRedBin( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_BlueRedBin<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_BlueGreenRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_BlueGreenRed<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_Grayscale( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_Grayscale<float>(dataIn[ n ].Distance);
    }
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void ColorizeRayData::Colorize_ShiftGray( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift )
{
    for ( unsigned int n = 0U; n < dataSize; n++)
    {
        dataOut[n] = ColorizeNumber_ShiftGray<float>(dataIn[ n ].Distance, shift);
    }
}
