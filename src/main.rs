#[allow(unused)]
mod renderer;
mod json;
#[allow(unused)]
mod gltf;
mod base64;

use nalgebra::*;

use renderer::StaticMesh;
use windows::{core::*, Win32::System::LibraryLoader::*, Win32::Foundation::*, Win32::UI::WindowsAndMessaging::*, Win32::UI::Input::KeyboardAndMouse::*, Win32::UI::Input::*};

fn main() -> std::result::Result<(), String> {
  unsafe {
    let raw_input_device = RAWINPUTDEVICE {
      usUsagePage: 0x01,
      usUsage: 0x02,
      dwFlags: RIDEV_NOLEGACY,
      ..Default::default()
    };

    RegisterRawInputDevices(&[raw_input_device], std::mem::size_of_val(&raw_input_device) as u32).map_err(|_|"failed to register mouse")?;

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
    
    let mut events = WindowEvents::default();
    
    SetWindowLongPtrA(window, GWLP_USERDATA, &mut events as *mut WindowEvents as isize);
    
    let mut renderer = renderer::Renderer::new(window)?;
    
    // Load the model
    let gltf_contents = gltf::load("models/bistro/scene.gltf")?;

    // Load all meshes onto device
    let mut meshes = Vec::<Vec<renderer::StaticMesh>>::new();
    for cpu_mesh in gltf_contents.meshes.iter() {
      let v = cpu_mesh.iter().map(|cpu_mesh|renderer.new_static_mesh(cpu_mesh)).collect();
      meshes.push(v);
    }

    std::assert!(meshes.len() == gltf_contents.meshes.len());

    let mut shit = Vec::<(renderer::Transform, Matrix4<f32>, StaticMesh)>::new();

    {
      let mut stack = Vec::<(Matrix4<f32>, usize)>::new();

      for n in &gltf_contents.scenes[gltf_contents.root_scene] {
        stack.push((Matrix4::<f32>::identity(), *n));
      }

      while let Some((parent_transform, node_idx)) = stack.pop() {
        let node = &gltf_contents.nodes[node_idx];

        let transform = parent_transform * node.local_transform;

        if let Some(mesh_idx) = node.mesh {
          let meshes = &meshes[mesh_idx];
          for m in meshes {
            shit.push((renderer.new_transform(), transform, *m));
          }
        }

        for c in &node.children {
          stack.push((transform, *c));
        }
      }
    }

    // Prepare for rendering
    let mut render_queue = Vec::new();

    let mut last_time = std::time::Instant::now();

    let camera_speed: f32 = 5.0;

    let mut camera_pitch = 0.0;
    let mut camera_yaw = 0.0;
    let mut camera_pos = Vector3::new(-5.0, 2.0, 0.0);

    let look_sensitivity = 0.001;
    
    // Main loop
    loop {
      let now = std::time::Instant::now();
      let delta_time = (now - last_time).as_secs_f32();
      last_time = now;

      events.reset();
      let mut msg = MSG::default();
      
      while PeekMessageA(&mut msg, window, 0, 0, PM_REMOVE).into() {
        _ = TranslateMessage(&msg);
        DispatchMessageA(&msg);
      }
      
      if events.closed {
        break;
      }

      camera_pitch -= events.mouse_dy * look_sensitivity;
      camera_yaw -= events.mouse_dx * look_sensitivity;

      let camera_rotation = Matrix4::from_euler_angles(camera_pitch, camera_yaw, 0.0);

      let forward = (camera_rotation * Vector4::new(0.0, 0.0, -1.0, 0.0)).xyz();
      let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0));

      if keydown(VK_W) {
        camera_pos += forward * camera_speed * delta_time;
      }

      if keydown(VK_S) {
        camera_pos -= forward * camera_speed * delta_time;
      }

      if keydown(VK_A) {
        camera_pos -= right * camera_speed * delta_time;
      }                                                                                     
                                                                                            
      if keydown(VK_D) {                                                                    
        camera_pos += right * camera_speed * delta_time;
      }

      renderer.acquire_frame();
      
      render_queue.clear();
      
      for (t, matrix, mesh) in &shit {
        renderer.write_transform(*t, matrix);
        render_queue.push((*mesh, *t));
      }


      let camera_transform = Matrix4::new_translation(&camera_pos) * camera_rotation;
      let view_matrix = camera_transform.try_inverse().expect("Failed to inverse camera matrix");
      renderer.render(&render_queue, view_matrix);
    }
    
    return Ok(());
  }
}

fn keydown(key: VIRTUAL_KEY) -> bool {
  return (unsafe{GetKeyState(key.0 as i32)} as u16 & 0x8000) != 0;
}

struct WindowEvents {
  closed: bool,
  mouse_dx: f32,
  mouse_dy: f32
}

impl WindowEvents {
  fn default() -> Self {
    return WindowEvents {
      closed: false,
      mouse_dx: 0.0,
      mouse_dy: 0.0
    };
  }

  fn reset(&mut self) {
    *self = Self::default();
  }
}

extern "system" fn window_proc(window: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
  let events: &mut WindowEvents = unsafe{&mut *(GetWindowLongPtrA(window, GWLP_USERDATA) as *mut WindowEvents)};
  
  match msg {
    WM_CLOSE => {
      events.closed = true;
    },

    WM_INPUT => {
      unsafe {
        let mut data_size: u32 = 0;

        let raw_input = HRAWINPUT(lparam.0 as *mut LPARAM as *mut std::ffi::c_void);
        GetRawInputData(raw_input, RID_INPUT, None, &mut data_size, std::mem::size_of::<RAWINPUTHEADER>() as u32);

        let data = vec![0 as u8;data_size as usize];
        GetRawInputData(raw_input, RID_INPUT, Some(data.as_ptr() as _), &mut data_size, std::mem::size_of::<RAWINPUTHEADER>() as u32);

        let input = &*(data.as_ptr() as *const RAWINPUT);

        if input.header.dwType == RIM_TYPEMOUSE.0 {
          events.mouse_dx += input.data.mouse.lLastX as f32;
          events.mouse_dy += input.data.mouse.lLastY as f32;
        }
      }
    }

    _ => {
    }
  }
  
  return unsafe{DefWindowProcA(window, msg, wparam, lparam)};
}