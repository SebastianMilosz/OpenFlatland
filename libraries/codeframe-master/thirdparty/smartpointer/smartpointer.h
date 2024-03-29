#ifndef SERIALIZABLE_SMART_PTR_HPP_INCLUDED
#define SERIALIZABLE_SMART_PTR_HPP_INCLUDED

#include <memory>

#define smart_ptr std::shared_ptr

#define smart_ptr_wild smart_ptr

//#define smart_isValid(ptr) (ptr.IsValid())
#define smart_ptr_isValid(ptr) (ptr != NULL)

//#define smart_ptr_getRaw(ptr) (ptr)
#define smart_ptr_getRaw(ptr) (ptr.get())

//#define smart_ptr_getCount(ptr) (ptr.GetCount())
#define smart_ptr_getCount(ptr) (ptr.use_count())

#define smart_dynamic_pointer_cast std::dynamic_pointer_cast

#endif // SERIALIZABLE_SMART_PTR_HPP_INCLUDED
