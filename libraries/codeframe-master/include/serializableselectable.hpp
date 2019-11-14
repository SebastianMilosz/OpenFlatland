#ifndef SERIALIZABLESELECTABLE_HPP_INCLUDED
#define SERIALIZABLESELECTABLE_HPP_INCLUDED

#include <sigslot.h>
#include <smartpointer.h>

using namespace sigslot;

namespace codeframe
{
    class cSerializableContainer;
    class ObjectNode;

    /*****************************************************************************/
    /**
      * @brief
      * @author Sebastian Milosz
      * @version 1.0
      * @note cSerializableSelectable
     **
    ******************************************************************************/
    class cSerializableSelectable
    {
        friend class ObjectContainer;

        public:
                     cSerializableSelectable( ObjectNode& sint );
            virtual ~cSerializableSelectable();

            virtual void Select( bool state = true );
            virtual bool IsSelected();

            signal1< smart_ptr<ObjectNode> > signalSelectionChanged;

        protected:
            template<class containerClass>
            void ConectToContainer( containerClass* containerObject, smart_ptr<ObjectNode> sthis )
            {
                m_smartThis = sthis;
                signalSelectionChanged.connect( containerObject, &containerClass::slotSelectionChanged );
            }

            void DisconectFromContainer();

        private:
            bool m_selected;
            smart_ptr<ObjectNode> m_smartThis;
    };
}

#endif // SERIALIZABLESELECTABLE_HPP_INCLUDED
