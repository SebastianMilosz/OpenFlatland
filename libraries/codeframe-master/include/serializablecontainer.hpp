#ifndef CSERIALIZABLECONTAINER_H
#define CSERIALIZABLECONTAINER_H

#include <exception>
#include <stdexcept>
#include <vector>
#include <cstdbool>

#include <MathUtilities.h>
#include <smartpointer.h>

#include "serializable.hpp"
#include "propertyignorelist.hpp"

#define MAXID 100

namespace codeframe
{
    /*****************************************************************************/
    /**
      * @brief
     **
    ******************************************************************************/
    class cSerializableContainer : public cSerializable
    {
        friend class cSerializableSelectable;

        CODEFRAME_META_CLASS_NAME( "cSerializableContainer" );
        CODEFRAME_META_BUILD_ROLE( codeframe::CONTAINER  );
        CODEFRAME_META_BUILD_TYPE( codeframe::STATIC );

        public:
                     cSerializableContainer( const std::string& name, cSerializableInterface* parentObject );
            virtual ~cSerializableContainer();

            virtual smart_ptr<cSerializableInterface> Create(
                                                             const std::string& className,
                                                             const std::string& objName,
                                                             const std::vector<codeframe::VariantValue>& params = std::vector<codeframe::VariantValue>()
                                                             ) = 0;

            smart_ptr<cSerializableInterface> operator[]( int i );

            virtual void CreateRange( const std::string& className, const std::string& objName, int range );
            virtual bool Dispose( unsigned int id );
            virtual bool Dispose( const std::string& objName );
            virtual bool Dispose( smart_ptr<cSerializableInterface> obj );
            virtual bool DisposeByBuildType( eBuildType serType, cIgnoreList ignore = cIgnoreList() );
            virtual bool Dispose();

            int         Count() const;
            bool        IsName( const std::string& name );
            std::string CreateUniqueName( const std::string& nameBase );
            bool        IsInRange( unsigned int cnt ) const;
            bool        Select( int pos );
            bool        IsSelected();

            smart_ptr<cSerializableInterface> GetSelected();
            smart_ptr<cSerializableInterface> Get( int id );

            int Add( smart_ptr<cSerializable> classType, int pos = -1 );

            signal1< smart_ptr<cSerializableInterface> > signalContainerSelectionChanged;

        protected:
            virtual int InsertObject( smart_ptr<cSerializable> classType, int pos = -1 );

        private:
            void slotSelectionChanged( smart_ptr<cSerializableInterface> obj );

            std::vector< smart_ptr<cSerializable> > m_containerVector;
            smart_ptr<cSerializableInterface> m_selected;

            unsigned int m_size;
    };

}

#endif // CSERIALIZABLECONTAINER_H
