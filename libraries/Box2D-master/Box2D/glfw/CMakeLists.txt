
#define the glfw files.
set(glfw_SRCS
clipboard.c
context.c
egl_context.c
gamma.c
glx_context.c
init.c
input.c
joystick.c
monitor.c
time.c
wgl_context.c
config.h
egl_platform.h
glext.h
glfw3.h
glfw3native.h
glx_platform.h
glxext.h
internal.h
nsgl_platform.h
wgl_platform.h
wglext.h
nsgl_context.m
)

#define the cocoa files.
set(cocoa_SRCS
cocoa_gamma.c
cocoa_time.c
cocoa_clipboard.m
cocoa_init.m
cocoa_joystick.m
cocoa_monitor.m
cocoa_window.m
cocoa_platform.h
)

#define the x11 files.
set(x11_SRCS
x11_clipboard.c
x11_gamma.c
x11_init.c
x11_joystick.c
x11_monitor.c
x11_time.c
x11_unicode.c
x11_window.c
x11_platform.h
)

#define the win32 glfw files.
set(win32_glfw_SRCS
win32_clipboard.c
win32_gamma.c
win32_init.c
win32_joystick.c
win32_monitor.c
win32_time.c
win32_window.c
window.c
win32_platform.h
)