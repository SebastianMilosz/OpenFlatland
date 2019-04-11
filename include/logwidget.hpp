#ifndef LOGWIDGET_HPP
#define LOGWIDGET_HPP

#include <imgui.h>
#include <imgui-SFML.h>
#include <utilities/LoggerUtilities.h>
#include <utilities/FileUtilities.h>

class LogWidget : public sigslot::has_slots<>
{
    public:
        LogWidget();
       ~LogWidget();

        void Clear();
        void AddLog(const char* fmt, ...) IM_FMTARGS(2);
        void Draw(const char* title, bool* p_open = NULL);
        void OnLogMessage(const std::string& timestamp, const std::string& title, const std::string& msg, int type);

    private:
        ImGuiTextBuffer     Buf;
        ImGuiTextFilter     Filter;
        ImVector<int>       LineOffsets;        // Index to lines offset
        bool                ScrollToBottom;
};

#endif // LOGWIDGET_HPP
