#ifndef QTSERIALIZABLEUTILITIES_H
#define QTSERIALIZABLEUTILITIES_H

#include "serializable.h"

#include <iostream>       // std::cerr
#include <stdexcept>      // std::out_of_range
#include <QComboBox>

namespace codeframe
{

    class qtSerializableUtilities
    {
        public:
            static void FillQComboBox( QComboBox* cmb, Property& prop )
            {
                if( cmb )
                {
                    cmb->blockSignals(true);
                    if( prop.Info().GetKind() == KIND_ENUM)
                    {
                        std::string enumString = prop.Info().GetEnum();

                        if( enumString.size() )
                        {
                            QStringList arrDiet;
                            std::stringstream ss( enumString );
                            std::string token;

                            while(std::getline(ss, token, ',')) { arrDiet << token.c_str(); }

                            cmb->insertItems(0, arrDiet);
                        }
                        else
                        {

                        }

                        cmb->setCurrentIndex(-1);
                    }
                    cmb->blockSignals(false);
                }
            }
    };

} // codeframe

#endif // QTSERIALIZABLEUTILITIES_H

