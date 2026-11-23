#ifndef SELECTABLE_ELEMENT_HPP_INCLUDED
#define SELECTABLE_ELEMENT_HPP_INCLUDED

#include <cstdint>

/*****************************************************************************/
/**
  * @brief Interface for drawable elements made of nodes that can be hovered
  *        with the mouse. Node id NO_NODE (-1) means "nothing hovered".
 **
******************************************************************************/
class SelectableElement
{
    public:
        static constexpr int32_t NO_NODE = -1;

        virtual ~SelectableElement() = default;

        virtual void OnHover(int32_t nodeId) = 0;
};

#endif // SELECTABLE_ELEMENT_HPP_INCLUDED
