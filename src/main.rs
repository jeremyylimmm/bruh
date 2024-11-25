#[allow(unused)]
mod renderer;
mod json;
#[allow(unused)]
mod gltf;
mod base64;

use nalgebra::*;
use windows::{core::*, Win32::System::LibraryLoader::*, Win32::Foundation::*, Win32::UI::WindowsAndMessaging::*, Win32::UI::Input::KeyboardAndMouse::*, Win32::UI::Input::*};

use bevy_ecs::prelude::*;

#[derive(Component)]
struct StaticMeshComponent {
  mesh: renderer::StaticMesh,
}

#[derive(Component)]
struct TransformComponent {
  transform: renderer::Transform,
  matrix: Matrix4<f32>
}

#[derive(Component)]
struct ChildrenComponent {
  children: Vec<Entity>,
}

#[derive(Component)]
struct ParentComponent {
  parent: Entity
}

#[derive(Component)]
struct LocalTransformComponent {
  matrix: Matrix4<f32>
}

fn parent_entity(world: &mut World, child: Entity, parent: Entity) {
  if let Some(old_parent_comp) = world.get::<ParentComponent>(child) {
    world.get_entity_mut(old_parent_comp.parent).unwrap().get_mut::<ChildrenComponent>().unwrap().children.retain(|x|*x != child);
  }

  world.entity_mut(child).insert(ParentComponent{parent});

  if !world.entity(parent).contains::<ChildrenComponent>() {
    world.entity_mut(parent).insert(ChildrenComponent{children:Vec::new()});
  }

  world.get_mut::<ChildrenComponent>(parent).unwrap().children.push(child);
}

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

    let mut world = World::new();

    {
      let mut stack = Vec::<(Option<Entity>, usize)>::new();

      for n in &gltf_contents.scenes[gltf_contents.root_scene] {
        stack.push((None, *n));
      }

      while let Some((parent, node_idx)) = stack.pop() {
        let node = &gltf_contents.nodes[node_idx];

        let e = world.spawn((
          TransformComponent{transform:renderer.new_transform(), matrix:Matrix4::identity()},
          LocalTransformComponent{matrix:node.local_transform}
        )).id();

        if let Some(p) = parent {
          parent_entity(&mut world, e, p);
        }

        if let Some(mesh_idx) = node.mesh {
          for m in &meshes[mesh_idx] {
            let me = world.spawn((
              TransformComponent{transform:renderer.new_transform(), matrix:Matrix4::identity()},
              LocalTransformComponent{matrix:Matrix4::identity()},
              StaticMeshComponent{mesh: *m}
            )).id();

            parent_entity(&mut world, me, e);
          }
        }

        for c in &node.children {
          stack.push((Some(e), *c));
        }
      }
    }

    // Resolve transforms

    {
      let mut stack = Vec::<(Matrix4::<f32>, Entity)>::new();

      for (mut global, local, children) in world.query_filtered::<(&mut TransformComponent, &mut LocalTransformComponent, Option<&ChildrenComponent>), Without<ParentComponent>>().iter_mut(&mut world) {
        global.matrix = local.matrix;

        if let Some(children_comp) = children {
          for c in &children_comp.children {
            stack.push((global.matrix, *c));
          }
        }
      }

      while let Some((parent_transform, e)) = stack.pop() {
        let local = world.get::<LocalTransformComponent>(e).unwrap().matrix;
        let mut global = world.get_mut::<TransformComponent>(e).unwrap();

        let transform = parent_transform * local;
        global.matrix = transform;

        if let Some(children) = world.get::<ChildrenComponent>(e) {
          for c in &children.children {
            stack.push((transform, *c));
          }
        }
      }
    }

    // Prepare for rendering
    let mut render_queue = Vec::new();

    let mut last_time = std::time::Instant::now();

    let camera_speed: f32 = 10.0;

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
      
      for (mesh, transform) in world.query::<(&StaticMeshComponent, &TransformComponent)>().iter(&world) {
        renderer.write_transform(transform.transform, &transform.matrix);
        render_queue.push((mesh.mesh, transform.transform));
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