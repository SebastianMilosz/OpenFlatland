#ifndef COLORIZE_RAY_DATA_HPP
#define COLORIZE_RAY_DATA_HPP

#include <SFML/Graphics.hpp>
#include <thrust/device_vector.h>

#include "entity_vision_node.hpp"

class ColorizeRayData
{
    public:
        enum eColorizeMode
        {
            IronBow = 0,
            RedYellow,
            BlueRed,
            BlackRed,
            BlueRedBin,
            BlueGreenRed,
            Grayscale,
            ShiftGray
        };

        void Colorize( const eColorizeMode mode, const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );

    public:
        void Colorize_IronBow     ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_RedYellow   ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRed     ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlackRed    ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRedBin  ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueGreenRed( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_Grayscale   ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_ShiftGray   ( const thrust::host_vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift );
};

#endif // COLORIZE_RAY_DATA_HPP
