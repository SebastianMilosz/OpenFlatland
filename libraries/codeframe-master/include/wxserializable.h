#ifndef WXSERIALIZABLE_H
#define WXSERIALIZABLE_H

#include "serializable.h"

namespace codeframe
{
    class wxPropertyGrid;

    class wxSerializable : public cSerializable
    {
        public:
            wxSerializable( std::string name, cSerializable* parent = NULL );

            void wxSetPropertyGrid( wxPropertyGrid* propgrid );
            void wxGetPropertyGrid( wxPropertyGrid* propgrid, bool changedFilter = true );
            void wxUpdatePropertyValue( Property* prop );

        private:
            wxPropertyGrid* m_wxPropertyGridPtr;
    };
}

#endif // WXSERIALIZABLE_H
