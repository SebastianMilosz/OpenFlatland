#ifndef LOGWIDGET_HPP
#define LOGWIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <serializable.hpp>
#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>

using namespace codeframe;

class ConsoleWidget : public sigslot::has_slots<>
{
    public:
        ConsoleWidget( cSerializableInterface& parent );
       ~ConsoleWidget();

        void Clear();
        void Draw(const char* title, bool* p_open = NULL);
        void OnLogMessage(const std::string& timestamp, const std::string& title, const std::string& msg, int type);

    private:
        // Portable helpers
        static int   Stricmp(const char* str1, const char* str2)         { int d; while ((d = toupper(*str2) - toupper(*str1)) == 0 && *str1) { str1++; str2++; } return d; }
        static int   Strnicmp(const char* str1, const char* str2, int n) { int d = 0; while (n > 0 && (d = toupper(*str2) - toupper(*str1)) == 0 && *str1) { str1++; str2++; n--; } return d; }
        static char* Strdup(const char *str)                             { size_t len = strlen(str) + 1; void* buf = malloc(len); IM_ASSERT(buf); return (char*)memcpy(buf, (const void*)str, len); }
        static void  Strtrim(char* str)                                  { char* str_end = str + strlen(str); while (str_end > str && str_end[-1] == ' ') str_end--; *str_end = 0; }

        void AddLog(const char* fmt, ...) IM_FMTARGS(2);
        static int TextEditCallbackStub( ImGuiInputTextCallbackData* data );
        int TextEditCallback( ImGuiInputTextCallbackData* data );

        cSerializableInterface& m_parent;
        ImGuiTextBuffer         m_Buf;
        ImGuiTextFilter         m_Filter;
        ImVector<int>           m_LineOffsets;        // Index to lines offset
        bool                    m_ScrollToBottom;
        char                    m_InputBuf[256];
        ImVector<const char*>   m_Commands;
        ImVector<char*>         m_History;
        int                     m_HistoryPos;    // -1: new line, 0..History.Size-1 browsing history.
};

#endif // LOGWIDGET_HPP
