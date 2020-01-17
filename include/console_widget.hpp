#ifndef LOGWIDGET_HPP
#define LOGWIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <serializable_object.hpp>
#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>
#include <utilities/DataTypesUtilities.h>

using namespace codeframe;

class ConsoleWidget : public sigslot::has_slots<>
{
    template<uint32_t S, typename T>
    using CircularBuffer = utilities::data::CircularBuffer<S,T>;

    public:
        ConsoleWidget( ObjectNode& parent );
       ~ConsoleWidget() = default;

        void Clear();
        void Draw(const char* title, bool* p_open = NULL);
        void OnLogMessage(const std::string& timestamp, const std::string& title, const std::string& msg, int type);
        void Save( utilities::data::DataStorage& ds );
        void Load( utilities::data::DataStorage& ds );

    private:
        void AddLog(const char* fmt, ...) IM_FMTARGS(2);
        static int TextEditCallbackStub( ImGuiInputTextCallbackData* data );
        int TextEditCallback( ImGuiInputTextCallbackData* data );

        ObjectNode&                     m_parent;
        ImGuiTextBuffer                 m_Buf;
        ImGuiTextFilter                 m_Filter;
        ImVector<int>                   m_LineOffsets;
        bool                            m_ScrollToBottom;
        char                            m_InputBuf[256];
        ImVector<const char*>           m_Commands;
        CircularBuffer<32, std::string> m_History;
};

#endif // LOGWIDGET_HPP
