#ifndef SERIALIZABLE_CONTAINER_H
#define SERIALIZABLE_CONTAINER_H

#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdbool>

#include <MathUtilities.h>
#include <smartpointer.h>

#include "serializable_object.hpp"
#include "propertyignorelist.hpp"

#define MAXID 100

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class ObjectContainer : public Object
    {
        friend class cSelectable;

        CODEFRAME_META_CLASS_NAME( "ObjectContainer" );
        CODEFRAME_META_BUILD_ROLE( codeframe::CONTAINER  );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
                     ObjectContainer( const std::string& name, ObjectNode* parentObject = nullptr );
                     ObjectContainer( const std::string& name, smart_ptr<ObjectNode> parentObject );
            virtual ~ObjectContainer();

            unsigned int Count() const override;

            smart_ptr<ObjectSelection> operator[]( const unsigned int i );
            smart_ptr<ObjectSelection> operator[]( const std::string& name );

            smart_ptr<ObjectSelection> Child( const unsigned int i ) override;
            smart_ptr<ObjectSelection> Child( const std::string& name ) override;

            virtual void CreateRange( const std::string& className, const std::string& objName, const int range );
            virtual bool Dispose( const unsigned int id );
            virtual bool Dispose( const std::string& objName );
            virtual bool Dispose( smart_ptr<ObjectNode> obj );
            virtual bool DisposeByBuildType( const eBuildType buildType, const cIgnoreList ignoreList = cIgnoreList() );
            virtual bool Dispose();

            bool         IsName( const std::string& name );
            std::string  CreateUniqueName( const std::string& nameBase );
            bool         IsInRange( const unsigned int cnt ) const;
            bool         Select( const int pos );
            bool         IsSelected();

            smart_ptr<ObjectNode> GetSelected();
            smart_ptr<ObjectNode> Get( const int id );

            int Add( smart_ptr<Object> classType, const int pos = -1 );

            signal1< smart_ptr<ObjectNode> > signalContainerSelectionChanged;

        protected:
            virtual int InsertObject( smart_ptr<Object> classType, const int pos = -1 );

        private:
            void slotSelectionChanged( smart_ptr<ObjectNode> obj );

            std::vector< smart_ptr<Object> > m_containerVector;
            smart_ptr<ObjectNode> m_selected;
    };

}

#endif // SERIALIZABLE_CONTAINER_H
