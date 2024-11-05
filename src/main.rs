mod renderer;
mod matrix;
mod json;

use windows::{core::*, Win32::System::LibraryLoader::*, Win32::Foundation::*, Win32::UI::WindowsAndMessaging::*};

fn main() -> std::result::Result<(), String> {
    let j = "{\n\
        \"first_name\": \"John\",\n\
        \"last_name\": \"Smith\",\n\
        \"is_alive\": true,\n\
        \"age\": 27,\n\
        \"address\": {\n\
            \"street_address\": \"21 2nd Street\",\n\
            \"city\": \"New York\",\n\
            \"state\": \"NY\",\n\
            \"postal_code\": \"10021-3100\"\n\
        },\n\
        \"phone_numbers\": [\n\
            {\n\
            \"type\": \"home\",\n\
            \"number\": \"212 555-1234\"\n\
            },\n\
            {\n\
            \"type\": \"office\",\n\
            \"number\": \"646 555-4567\"\n\
            }\n\
        ],\n\
        \"children\": [\n\
            \"Catherine\",\n\
            \"Thomas\",\n\
            \"Trevor\"\n\
        ],\n\
        \"spouse\": null\n\
    }";

    let node = json::parse(&j.to_string())?;
    println!("{:?}", node);

    unsafe {
        let wc = WNDCLASSA {
            hInstance: GetModuleHandleA(None).unwrap().into(),
            lpfnWndProc: Some(window_proc),
            lpszClassName: s!("bruh"),
            ..Default::default()
        };

        RegisterClassA(&wc as _);

        let window = CreateWindowExA(
            WINDOW_EX_STYLE::default(),
            wc.lpszClassName,
            s!("Bruh"),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            None,
            None,
            wc.hInstance,
            None
        ).unwrap();

        _ = ShowWindow(window, SW_MAXIMIZE);

        let mut events = WindowEvents {
            closed: false
        };

        SetWindowLongPtrA(window, GWLP_USERDATA, &mut events as *mut WindowEvents as isize);

        let mut renderer = renderer::Renderer::new(window)?;

        let start = std::time::Instant::now();

        loop {
            let mut msg = MSG::default();

            while PeekMessageA(&mut msg, window, 0, 0, PM_REMOVE).into() {
                _ = TranslateMessage(&msg);
                DispatchMessageA(&msg);
            }

            if events.closed {
                break;
            }

            renderer.render(start.elapsed().as_secs_f32());
        }

        return Ok(());
    }
}

struct WindowEvents {
    closed: bool
}

extern "system" fn window_proc(window: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let mut result = LRESULT::default();

    let events: &mut WindowEvents = unsafe{&mut *(GetWindowLongPtrA(window, GWLP_USERDATA) as *mut WindowEvents)};

    match msg {
        WM_CLOSE => {
            events.closed = true;
        },
        _ => {
            result = unsafe{DefWindowProcA(window, msg, wparam, lparam)};
        }
    }

    return result;
}