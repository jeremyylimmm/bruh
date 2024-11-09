mod renderer;
#[allow(unused)]
mod matrix;
mod json;
mod gltf;
mod base64;

use windows::{core::*, Win32::System::LibraryLoader::*, Win32::Foundation::*, Win32::UI::WindowsAndMessaging::*};

fn main() -> std::result::Result<(), String> {
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

        let cpu_meshes = gltf::load("models/flight_helmet/scene.gltf")?;
        let meshes = cpu_meshes.iter().map(|x|{
            (
                renderer.new_static_mesh(x),
                matrix::translation(&[0.0, -1.0, 0.0]) * matrix::scaling(&[2.0, 2.0, 2.0])
            )
        }).collect();

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

            renderer.render(&meshes, start.elapsed().as_secs_f32());
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