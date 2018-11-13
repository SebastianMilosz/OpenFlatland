#ifndef COLORIZEREALNBR_HPP_INCLUDED
#define COLORIZEREALNBR_HPP_INCLUDED

#include <SFML/Graphics.hpp>

class ColorizeRealNumbers
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

    void Colorize( eColorizeMode mode, const float* dataIn, sf::Color* dataOut, unsigned int dataSize );

    public:
        void Colorize_IronBow     ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_RedYellow   ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRed     ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlackRed    ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRedBin  ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueGreenRed( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_Grayscale   ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_ShiftGray   ( const float* dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift );
};

#endif // COLORIZEREALNBR_HPP_INCLUDED
