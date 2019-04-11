#include "logwidget.hpp"

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
LogWidget::LogWidget()
{
    //ctor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
LogWidget::~LogWidget()
{
    //dtor
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void LogWidget::OnLogMessage(const std::string& timestamp, const std::string& title, const std::string& msg, int type)
{
    AddLog( "[%s] : [%s] : [%d] : %s\n", timestamp.c_str(), title.c_str(), type, msg.c_str() );
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void LogWidget::Draw( const char* title, bool* p_open )
{
    ImGui::SetNextWindowSize(ImVec2(500,400), ImGuiCond_FirstUseEver);
    ImGui::Begin(title, p_open);
    if (ImGui::Button("Clear")) Clear();
    ImGui::SameLine();
    bool copy = ImGui::Button("Copy");
    ImGui::SameLine();
    Filter.Draw("Filter", -100.0f);
    ImGui::Separator();
    ImGui::BeginChild("scrolling", ImVec2(0,0), false, ImGuiWindowFlags_HorizontalScrollbar);
    if (copy) ImGui::LogToClipboard();

    if (Filter.IsActive())
    {
        const char* buf_begin = Buf.begin();
        const char* line = buf_begin;
        for (int line_no = 0; line != NULL; line_no++)
        {
            const char* line_end = (line_no < LineOffsets.Size) ? buf_begin + LineOffsets[line_no] : NULL;
            if (Filter.PassFilter(line, line_end))
                ImGui::TextUnformatted(line, line_end);
            line = line_end && line_end[1] ? line_end + 1 : NULL;
        }
    }
    else
    {
        ImGui::TextUnformatted(Buf.begin());
    }

    if (ScrollToBottom)
        ImGui::SetScrollHere(1.0f);
    ScrollToBottom = false;
    ImGui::EndChild();
    ImGui::End();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void LogWidget::Clear()
{
    Buf.clear();
    LineOffsets.clear();
}

/*****************************************************************************/
/**
  * @brief
 **
******************************************************************************/
void LogWidget::AddLog(const char* fmt, ...)
{
    int old_size = Buf.size();
    va_list args;
    va_start(args, fmt);
    Buf.appendfv(fmt, args);
    va_end(args);
    for (int new_size = Buf.size(); old_size < new_size; old_size++)
        if (Buf[old_size] == '\n')
            LineOffsets.push_back(old_size);
    ScrollToBottom = true;
}
