#ifndef COLORIZEREALNBR_HPP_INCLUDED
#define COLORIZEREALNBR_HPP_INCLUDED

#include <SFML/Graphics.hpp>

class ColorizeRealNumbers
{
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

    void Colorize( eColorizeMode mode, const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );

    private:
        void Colorize_IronBow     ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_RedYellow   ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRed     ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlackRed    ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueRedBin  ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_BlueGreenRed( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_Grayscale   ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize );
        void Colorize_ShiftGray   ( const uint16_t* dataIn, sf::Color* dataOut, unsigned int dataSize, uint8_t shift );
};

#endif // COLORIZEREALNBR_HPP_INCLUDED
