#include "serializableregister.h"

namespace codeframe
{

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cRegister::cRegister()
    {
        m_registerMode      = 0;
        m_enable            = false;

        m_registerRead      = 0;
        m_registerSizeRead  = 0;
        m_cellOffsetRead    = 0;
        m_cellSizeRead      = 0;

        m_registerWrite     = 0;
        m_registerSizeWrite = 0;
        m_cellOffsetWrite   = 0;
        m_cellSizeWrite     = 0;

        m_bitMaskRead		= 0;
        m_bitMaskWrite		= 0;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    cRegister::cRegister( eREG_MODE mod, uint16_t reg, uint16_t regSize, uint16_t cellOffset, uint16_t cellSize, uint16_t bitMask )
    {
        Set( mod, reg, regSize, cellOffset, cellSize, bitMask );
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    void cRegister::Set( eREG_MODE mod, uint16_t reg, uint16_t regSize, uint16_t cellOffset, uint16_t cellSize, uint16_t bitMask )
    {
        if( mod == R || mod == RW )
        {
            m_registerRead      = reg;
            m_registerSizeRead  = regSize;
            m_cellOffsetRead    = cellOffset;
            m_cellSizeRead      = cellSize;
            m_bitMaskRead       = bitMask;

            m_registerMode |= 0x01;
        }

        if( mod == W || mod == RW )
        {
            m_registerWrite     = reg;
            m_registerSizeWrite = regSize;
            m_cellOffsetWrite   = cellOffset;
            m_cellSizeWrite     = cellSize;
            m_bitMaskWrite      = bitMask;

            m_registerMode |= 0x02;
        }

        m_enable = true;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    eREG_MODE cRegister::Mode() const
    {
        if( m_registerMode == 0x03)  return RW;
        if( m_registerMode == 0x01)  return R;
        if( m_registerMode == 0x02)  return W;

        return R;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cRegister::IsEnable() const
    {
        return m_enable;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cRegister::IsRead() const
    {
        if(m_enable)
        {
            if(m_registerMode & R) return true;
        }
        return false;
    }

    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    bool cRegister::IsWrite() const
    {
        if(m_enable)
        {
            if(m_registerMode & W) return true;
        }
        return false;
    }

}
