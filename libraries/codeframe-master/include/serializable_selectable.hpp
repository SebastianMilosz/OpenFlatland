#ifndef SELECTABLE_HPP_INCLUDED
#define SELECTABLE_HPP_INCLUDED

#include <sigslot.h>
#include <smartpointer.h>

using namespace sigslot;

namespace codeframe
{
    class ObjectNode;

    /*****************************************************************************/
    /**
      * @brief
      * @version 1.0
      * @note cSelectable
     **
    ******************************************************************************/
    class cSelectable
    {
        friend class ObjectContainer;

        public:
                     cSelectable( ObjectNode& sint );
            virtual ~cSelectable();

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

#endif // SELECTABLE_HPP_INCLUDED
