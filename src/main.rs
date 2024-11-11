mod renderer;
#[allow(unused)]
mod matrix;
mod json;
mod gltf;
mod base64;
mod ecs;

use windows::{core::*, Win32::System::LibraryLoader::*, Win32::Foundation::*, Win32::UI::WindowsAndMessaging::*};

struct StaticMeshComponent {
  mesh: renderer::StaticMesh
}

struct TransformComponent {
  matrix: matrix::Float4x4
}

impl ecs::Component for StaticMeshComponent {}
impl ecs::Component for TransformComponent {}

fn main() -> std::result::Result<(), String> {
  let mut world = ecs::World::new();
  
  world.register::<StaticMeshComponent>();
  world.register::<TransformComponent>();
  
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
    
    let mut ents = Vec::new();
    
    for cpu_mesh in gltf::load("models/flight_helmet/scene.gltf")? {
      let e = world.create();
      ents.push(e);
      
      world.add(e, StaticMeshComponent{
        mesh: renderer.new_static_mesh(&cpu_mesh)
      });
      
      let transform = matrix::translation(&[0.0, -1.0, 0.0]) * matrix::scaling(&[2.0, 2.0, 2.0]);
      
      world.add(e, TransformComponent{
        matrix: transform
      });
    }
    
    let start = std::time::Instant::now();
    
    let mut render_queue = Vec::new();
    
    loop {
      let mut msg = MSG::default();
      
      while PeekMessageA(&mut msg, window, 0, 0, PM_REMOVE).into() {
        _ = TranslateMessage(&msg);
        DispatchMessageA(&msg);
      }
      
      if events.closed {
        break;
      }
      
      render_queue.clear();
      
      for e in &ents {
        let mesh_comp = world.get::<StaticMeshComponent>(*e).unwrap();
        let transform_comp = world.get::<TransformComponent>(*e).unwrap();
        render_queue.push((mesh_comp.mesh, transform_comp.matrix));
      }
      
      renderer.render(&render_queue, start.elapsed().as_secs_f32());
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