#ifndef COLORIZE_RAY_DATA_HPP
#define COLORIZE_RAY_DATA_HPP

#include <SFML/Graphics.hpp>

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

        void Colorize( const eColorizeMode mode, const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );

    public:
        void Colorize_IronBow     ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_RedYellow   ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRed     ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlackRed    ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRedBin  ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueGreenRed( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_Grayscale   ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_ShiftGray   ( const std::vector<RayData>& dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift );
};

#endif // COLORIZE_RAY_DATA_HPP
