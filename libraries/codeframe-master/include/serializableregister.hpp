#ifndef SERIALIZABLEREGISTER_H
#define SERIALIZABLEREGISTER_H

#include <stdint.h>

namespace codeframe
{

    enum eREG_MODE { R = 0x01, W = 0x02, RW = 0x03 };

    #define To16b(n, m) (n | (m << 8))

    class cRegister
    {
        private:
           uint8_t  m_registerMode;
           bool     m_enable;

           uint16_t m_registerRead;
           uint16_t m_registerSizeRead;
           uint16_t m_cellOffsetRead;
           uint16_t m_cellSizeRead;
           uint16_t m_bitMaskRead;

           uint16_t m_registerWrite;
           uint16_t m_registerSizeWrite;
           uint16_t m_cellOffsetWrite;
           uint16_t m_cellSizeWrite;
           uint16_t m_bitMaskWrite;

        public:
            cRegister();
            cRegister( eREG_MODE mod, uint16_t reg, uint16_t regSize = 1,  uint16_t cellOffset = 0, uint16_t cellSize = 1, uint16_t bitMask = 0xFFFF );

            void Set( eREG_MODE mod, uint16_t reg, uint16_t regSize = 1, uint16_t cellOffset = 0, uint16_t cellSize = 1, uint16_t bitMask = 0xFFFF );

            bool      IsRead() const;
            bool      IsWrite() const;

            uint16_t  ReadRegister()        const { return m_registerRead;     }
            uint16_t  ReadRegisterSize()    const { return m_registerSizeRead; }
            uint16_t  ReadCellOffset()      const { return m_cellOffsetRead;   }
            uint16_t  ReadCellSize()        const { return m_cellSizeRead;     }
            uint16_t  ReadBitMask()         const { return m_bitMaskRead;      }

            uint16_t  WriteRegister()       const { return m_registerWrite;    }
            uint16_t  WriteRegisterSize()   const { return m_registerSizeWrite;}
            uint16_t  WriteCellOffset()     const { return m_cellOffsetWrite;  }
            uint16_t  WriteCellSize()       const { return m_cellSizeWrite;    }
            uint16_t  WriteBitMask()        const { return m_bitMaskWrite;     }

            eREG_MODE Mode() const;
            bool      IsEnable() const;

            // Operators
            cRegister& operator=(cRegister val)
            {
                m_registerMode      = val.m_registerMode;
                m_enable            = val.m_enable;

                m_registerRead      = val.m_registerRead;
                m_registerSizeRead	= val.m_registerSizeRead;
                m_cellOffsetRead	= val.m_cellOffsetRead;
                m_cellSizeRead      = val.m_cellSizeRead;
                m_bitMaskRead		= val.m_bitMaskRead;

                m_registerWrite     = val.m_registerWrite;
                m_registerSizeWrite	= val.m_registerSizeWrite;
                m_cellOffsetWrite	= val.m_cellOffsetWrite;
                m_cellSizeWrite     = val.m_cellSizeWrite;
                m_bitMaskWrite		= val.m_bitMaskWrite;

                return *this;
            }
    };

    class cRegisterFormatter
    {
        private:
            cRegister& m_reg;

        public:
            explicit cRegisterFormatter(cRegister& reg) : m_reg(reg)
            {
            }


    };

}

#endif // SERIALIZABLEREGISTER_H
