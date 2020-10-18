#ifndef COLORIZE_NUMBER_HPP_INCLUDED
#define COLORIZE_NUMBER_HPP_INCLUDED

#include <SFML/Graphics.hpp>

template<typename T>
sf::Color ColorizeNumber_IronBown(const T& value)
{
    sf::Color retColor;
    uint16_t data = value * 1000;

    retColor.r = data >> 6;
    if ( retColor.r > 255 ) retColor.r = 255;
    if ( data & 0x2000 )    retColor.g = 0x0ff & ( data >> 5 ); else retColor.g = 0;
    retColor.b = 0;


    if ( (data & 0x3800) == 0x0000 ) { retColor.b = 0x0ff & (( data >> 3 )); }
    if ( (data & 0x3800) == 0x0800 ) { retColor.b = 0x0ff & (( data >> 4 )); retColor.b = 255 - retColor.b; retColor.b = retColor.b + 127; }
    if ( (data & 0x3800) == 0x1000 ) { retColor.b = 0x0ff & (( data >> 4 )); retColor.b = 128 - retColor.b; }
    if ( (data & 0x3000) == 0x3000 ) { retColor.b = 0x0ff & (( data >> 4 )); }

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_RedYellow(const T& value)
{
    sf::Color retColor;
    uint16_t data = value * 1000;

    retColor.r = data >> 6;
    if (retColor.r > 255) retColor.r = 255;
    if (data & 0x2000) retColor.g = (0x0ff & (data >> 5)); else retColor.g = 0;
    retColor.b = 0;

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_BlueRed(const T& value)
{
    sf::Color retColor;
    uint16_t data = value * 1000;

    if( (data & 0x3000) == 0 )
    {
        retColor.g = 0;
        retColor.b = 0x07f & (data >> 5);
        retColor.b = retColor.b + 128;
        retColor.r = 0;
    }
    else if( (data & 0x3000) == 0x1000 )
    {
        retColor.b = 0x07f & (data >> 5);
        retColor.b = 255 - retColor.b;
        retColor.g = 0;
        retColor.r = 0x07f & (data >> 5);
    }
    else if( (data & 0x3000) == 0x2000 )
    {
        retColor.b = 0x07f & (data >> 5);
        retColor.b = 128 - retColor.b;
        retColor.r = 0x07f & (data >> 5);
        retColor.r = retColor.r + 128;
        retColor.g = 0;
    }
    else if((data & 0x3000) == 0x3000)
    {
        retColor.b = 0;
        retColor.r = 255;
        retColor.g = 0x0ff & (data >> 4);
    }

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_BlackRed(const T& value)
{
    sf::Color retColor;

    uint16_t data = value * 1000;

    if ((data & 0x3000) == 0)
    {
        retColor.g = 0;
        retColor.b = (data >> 4);
        retColor.b = retColor.b & 0xff;
        retColor.r = 0;
    }
    else if ((data & 0x3000) == 0x1000)
    {
        retColor.b = 0x07f & (data >> 5);
        retColor.b = 255 - retColor.b;
        retColor.g = 0;
        retColor.r = 0x07f & (data >> 5);
    }
    else if ((data & 0x3000) == 0x2000)
    {
        retColor.b = 0x07f & (data >> 5);
        retColor.b = 128 - retColor.b;
        retColor.r = 0x07f & (data >> 5);
        retColor.r = retColor.r + 128;
        retColor.g = 0;
    }
    else if ((data & 0x3000) == 0x3000)
    {
        retColor.b = 0;
        retColor.r = 255;
        retColor.g = 0x0ff & (data >> 4);
    }

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_BlueRedBin(const T& value)
{
    sf::Color retColor;

    uint16_t data = value * 1000;

    retColor.r = data >> 6;
    if (retColor.r > 255) retColor.r = 255;
    retColor.b = data >> 6;
    if (retColor.b > 255) retColor.b = 255;

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_BlueGreenRed(const T& value)
{
    sf::Color retColor;

    uint16_t data = value * 1000;

    if ((data & 0x3000) == 0)
    {
        retColor.b = 255;
        retColor.g = 0x0ff & (data >> 4);
        retColor.r = 0;
    }
    else if ((data & 0x3000) == 0x1000)
    {
        retColor.b = 0x0ff & (data >> 4);
        retColor.b = 255 - retColor.b;
        retColor.g = 255;
        retColor.r = 0;
    }
    else if ((data & 0x3000) == 0x2000)
    {
        retColor.b = 0;
        retColor.g = 255;
        retColor.r = 0x0ff & (data >> 4);
    }
    else if ((data & 0x3000) == 0x3000)
    {

        retColor.b = 0;
        retColor.g = 0x0ff & (data >> 4);
        retColor.g = 255 - retColor.g;
        retColor.r = 255;
    }

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_Grayscale(const T& value)
{
    sf::Color retColor;

    uint16_t data = value * 1000;

    retColor.r = data >> 6;
    retColor.g = retColor.r;
    retColor.b = retColor.r;

    return retColor;
}

template<typename T>
sf::Color ColorizeNumber_ShiftGray(const T& value, uint8_t shift)
{
    sf::Color retColor;

    uint16_t data = value * 1000;

    retColor.r = data >> shift;
    if (retColor.r > 255) retColor.r = 255;

    retColor.g = retColor.r;
    retColor.b = retColor.r;

    return retColor;
}

#endif // COLORIZE_NUMBER_HPP_INCLUDED
