mod renderer;
mod json;
mod gltf;
mod base64;
mod ecs;

use nalgebra::*;

use renderer::StaticMesh;
use windows::{core::*, Win32::System::LibraryLoader::*, Win32::Foundation::*, Win32::UI::WindowsAndMessaging::*, Win32::UI::Input::KeyboardAndMouse::*};

struct StaticMeshComponent {
  mesh: renderer::StaticMesh
}

struct LocalTransformComponent {
  matrix: Matrix4<f32>,
}

struct TransformComponent {
  matrix: Matrix4<f32>
}

struct ChildrenComponent {
  children: Vec<ecs::Entity>,
}

struct ParentComponent {
  parent: ecs::Entity
}

impl ecs::Component for StaticMeshComponent {}
impl ecs::Component for LocalTransformComponent {}
impl ecs::Component for TransformComponent {}
impl ecs::Component for ChildrenComponent {}
impl ecs::Component for ParentComponent {}

fn main() -> std::result::Result<(), String> {
  let mut world = ecs::World::new();
  
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
    
    // Load the model
    let gltf_contents = gltf::load("models/bistro/scene.gltf")?;

    // Load all meshes onto device
    let mut meshes = Vec::<renderer::StaticMesh>::new();
    for cpu_mesh in gltf_contents.meshes {
      meshes.push(renderer.new_static_mesh(&cpu_mesh));
    }

    let mut shit = Vec::<(Matrix4<f32>, StaticMesh)>::new();

    {
      let mut stack = Vec::<(Matrix4<f32>, usize)>::new();

      for n in &gltf_contents.scenes[gltf_contents.root_scene] {
        stack.push((Matrix4::<f32>::identity(), *n));
      }

      while let Some((parent_transform, node_idx)) = stack.pop() {
        let node = &gltf_contents.nodes[node_idx];

        let transform = parent_transform * node.local_transform;

        if let Some(m) = node.mesh {
          shit.push((transform, meshes[m]));
        }

        for c in &node.children {
          stack.push((transform, *c));
        }
      }
    }

    //{ // Construct the scene hierarchy
    //  let mut stack = Vec::<(ecs::Entity, usize)>::new();

    //  for n in &gltf_contents.scenes[gltf_contents.root_scene] {
    //    stack.push((ecs::null_entity(), *n));
    //  }

    //  while !stack.is_empty() {
    //    let (parent, n) = stack.pop().unwrap();
    //    let node = &gltf_contents.nodes[n];

    //    let e = world.create();
    //    world.add::<TransformComponent>(e, TransformComponent{matrix:matrix::Float4x4::identity()});
    //    world.add::<LocalTransformComponent>(e, LocalTransformComponent{matrix:node.local_transform});

    //    if !parent.is_null() {
    //      if !world.has::<ChildrenComponent>(parent) {
    //        world.add::<ChildrenComponent>(parent, ChildrenComponent{children: Vec::new()});
    //      }

    //      world.get_mut::<ChildrenComponent>(parent).unwrap().children.push(e);
    //      world.add::<ParentComponent>(e, ParentComponent{parent});
    //    }

    //    if let Some(m) = node.mesh {
    //      world.add::<StaticMeshComponent>(e, StaticMeshComponent{mesh:meshes[m]});
    //    }

    //    for c in &node.children {
    //      stack.push((
    //        e, *c
    //      ));
    //    }
    //  }
    //}

    //// Bake all transforms

    //{
    //  for (global, local, _) in world.view::<(&mut TransformComponent, ecs::With<LocalTransformComponent>)>() {
    //    global.matrix = local.matrix;
    //  }

    //  let mut stack = Vec::<(matrix::Float4x4, ecs::Entity)>::new();

    //  for (t, children, _) in world.view::<(&TransformComponent, ecs::With<ChildrenComponent>, ecs::Without<ParentComponent>)>() {
    //    for c in &children.children {
    //      stack.push((t.matrix, *c));
    //    }
    //  }

    //  while !stack.is_empty() {
    //    let (parent_transform, e) = stack.pop().unwrap();

    //    let local = world.get::<LocalTransformComponent>(e).unwrap().matrix;
    //    let transform = parent_transform * local;

    //    world.get_mut::<TransformComponent>(e).unwrap().matrix = transform;

    //    if let Some(children) = world.get::<ChildrenComponent>(e) {
    //      for c in &children.children {
    //        stack.push((transform, *c));
    //      }
    //    }
    //  }
    //}

    // Prepare for rendering
    let mut render_queue = Vec::new();

    let mut camera_transform = Matrix4::<f32>::new_translation(&Vector3::new(-5.0, 2.0, 0.0));

    let camera_speed: f32 = 1.0;

    let mut last_time = std::time::Instant::now();
    
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

      if keydown(VK_W) {
        camera_transform = Matrix4::new_translation(&Vector3::new(0.0, 0.0, -camera_speed * delta_time)) * camera_transform;
      }

      if keydown(VK_S) {
        camera_transform = Matrix4::new_translation(&Vector3::new(0.0, 0.0, camera_speed * delta_time)) * camera_transform;
      }

      if keydown(VK_A) {
        camera_transform = Matrix4::new_translation(&Vector3::new(-camera_speed * delta_time, 0.0, 0.0)) * camera_transform;
      }                                                                                     
                                                                                            
      if keydown(VK_D) {                                                                    
        camera_transform = Matrix4::new_translation(&Vector3::new( camera_speed * delta_time, 0.0, 0.0)) * camera_transform;
      }
      
      render_queue.clear();
      
      //for (m, t, _) in world.view::<(&StaticMeshComponent, ecs::With<TransformComponent>)>() {
      //  render_queue.push((m.mesh, t.matrix));
      //}

      for (t, m) in &shit {
        render_queue.push((*m, *t));
      }

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
}

impl WindowEvents {
  fn reset(&mut self) {
    *self = WindowEvents {
      closed: false,
    };
  }
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