use renderer_pool_handle_derive::HandledPoolHandle;
use windows::Win32::{Graphics::Direct3D::Fxc::*, Graphics::Direct3D::Dxc::*, Graphics::Direct3D12::*, Graphics::Direct3D::*, Graphics::Dxgi::Common::*, Graphics::Dxgi::*, Foundation::*, UI::WindowsAndMessaging::*};
use windows::core::*;

use std::convert::identity;
use std::io::*;

use nalgebra::*;

const FRAMES_IN_FLIGHT: usize = 2;
const LINE_VBUFFER_CAP: usize = 10 * 1024;

#[allow(dead_code)]
pub struct Renderer {
  frame: usize,
  device: ID3D12Device,
  queue: ID3D12CommandQueue,
  fence_val: u64,
  fence: ID3D12Fence,
  frame_fences: [u64;FRAMES_IN_FLIGHT],
  cmd_allocators: [ID3D12CommandAllocator;FRAMES_IN_FLIGHT],
  cmd_lists: [ID3D12GraphicsCommandList;FRAMES_IN_FLIGHT],
  swapchain_fences: [u64;DXGI_MAX_SWAP_CHAIN_BUFFERS as _],
  swapchain_rtvs: [D3D12_CPU_DESCRIPTOR_HANDLE;DXGI_MAX_SWAP_CHAIN_BUFFERS as _],
  swapchain_buffers: Vec<ID3D12Resource>,
  swapchain_w: u32,
  swapchain_h: u32,
  swapchain: IDXGISwapChain3,
  window: HWND,
  rtv_heap: DescriptorHeap,
  cbv_srv_uav_heap: DescriptorHeap,
  dsv_heap: DescriptorHeap,

  line_pipeline: Pipeline, 
  scene_pipeline: Pipeline,

  camera_cbuffer: BufferedBuffer<Matrix4<f32>>,
  depth_buffer: Option<ID3D12Resource>,
  dsv: DescriptorHandle,

  static_mesh_pool: HandledPool<StaticMeshData, StaticMesh>,
  transform_pool: HandledPool<TransformData, Transform>,

  free_transforms: Vec<Transform>,

  line_vbuffer_ptrs: [*mut LineVertex; FRAMES_IN_FLIGHT],
  line_vbuffers: [ID3D12Resource; FRAMES_IN_FLIGHT],

  resident_resources: Vec<ID3D12Resource>
}

#[derive(Copy, Clone, HandledPoolHandle)]
pub struct StaticMesh {
  index: u32,
  generation: u32
}

#[derive(Copy, Clone, HandledPoolHandle)]
pub struct Transform {
  index: u32,
  generation: u32
}

#[repr(C)]
pub struct Vertex {
  pub pos: [f32;3],
  pub norm: [f32;3],
  pub tex_coord: [f32;2],
}

pub fn pad_256(x: usize) -> usize {
  return (x + 255) & (!(255 as usize));
}

impl Renderer {
  pub fn new(window: HWND) -> std::result::Result<Renderer, String> {
    unsafe {
      if cfg!(debug_assertions) {
        let mut debug_opt = Option::<ID3D12Debug>::None;
        D3D12GetDebugInterface(&mut debug_opt).map_err(|_|"Failed to enable debug layer")?;

        match debug_opt {
          Some(debug) => {
            debug.EnableDebugLayer();
          },
          _ => {
            return Err("Failed to enable debug layer".into());
          }
        }

        println!("Enabled D3D12 debug layer.");
      }

      let mut device_opt = Option::<ID3D12Device>::None;
      D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device_opt).map_err(|_| "Device creation error")?;

      let device = match device_opt {
        Some(dev) => {dev},
        _ => { return Err("Failed to create device".into()); }
      };

      let mut resident_resources = Vec::new();

      if cfg!(debug_assertions) {
        let info_queue = device.cast::<ID3D12InfoQueue>().map_err(|_|"Failed to get debug layer info queue")?;

        info_queue.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE).unwrap();
        info_queue.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE).unwrap();
        info_queue.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE).unwrap();
      }

      let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
      };

      let queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc).map_err(|_|"Failed to create command queue")?;

      let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(|_|"Failed to create fence")?;

      let create_allocator = |_:usize| -> ID3D12CommandAllocator { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap()};
      let cmd_allocators = std::array::from_fn (create_allocator);

      let create_list = |i:usize| -> ID3D12GraphicsCommandList { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &cmd_allocators[i], None).unwrap() };
      let cmd_lists = std::array::from_fn(create_list);

      for x in &cmd_lists {
        x.Close().unwrap();
      }

      let factory: IDXGIFactory1 = CreateDXGIFactory1().map_err(|_|"Failed to create factory")?;

      let (ww, wh) = hwnd_size(window);

      let swapchain_desc = DXGI_SWAP_CHAIN_DESC {
        BufferDesc: DXGI_MODE_DESC {
          Width: ww as _,
          Height: wh as _,
          Format: DXGI_FORMAT_R8G8B8A8_UNORM,
          ..Default::default()
        },

        SampleDesc: DXGI_SAMPLE_DESC {
          Count: 1,
          ..Default::default()
        },

        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,

        BufferCount: 3,
        OutputWindow: window,
        Windowed: TRUE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,

        ..Default::default()
      };

      let mut swapchain_opt = Option::<IDXGISwapChain>::None;
      if factory.CreateSwapChain(&queue, &swapchain_desc, &mut swapchain_opt).is_err()  {
        return Err("Failed to create first swap chain".into());
      }

      let swapchain: IDXGISwapChain3 = match swapchain_opt {
        Some(sc) => { sc.cast::<IDXGISwapChain3>().map_err(|_|"Failed to create swapchain")? },
        _ => {
          return Err("Failed to create second swap chain".into());
        },
      };

      let mut rtv_heap=DescriptorHeap::new(&device, 1024, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;
      let mut cbv_srv_uav_heap = DescriptorHeap::new(&device, 1000000, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE)?;
      let mut dsv_heap = DescriptorHeap::new(&device, 1024, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE)?;

      let rtvs: [DescriptorHandle;DXGI_MAX_SWAP_CHAIN_BUFFERS as _] = std::array::from_fn(|_|rtv_heap.alloc());

      let line_pipeline_info = PipelineCreateInfo {
        vs_code: load_shader("shaders/line.vso")?,
        ps_code: load_shader("shaders/line.pso")?,

        rtv_formats: &[swapchain_desc.BufferDesc.Format],

        root_params: &[
          root_param_cbv(D3D12_SHADER_VISIBILITY_VERTEX, 0, 0),
          root_param_srv(D3D12_SHADER_VISIBILITY_VERTEX, 0, 0),
        ],
        
        primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
      };

      let scene_pipeline_info = PipelineCreateInfo {
        vs_code: load_shader("shaders/triangle.vso")?,
        ps_code: load_shader("shaders/triangle.pso")?,

        rtv_formats: &[swapchain_desc.BufferDesc.Format],

        root_params: &[
          root_param_cbv(D3D12_SHADER_VISIBILITY_VERTEX, 0, 0)
        ],
        
        primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
      };

      let line_pipeline= Pipeline::create(&device, &line_pipeline_info)?;
      let scene_pipeline= Pipeline::create(&device, &scene_pipeline_info)?;

      // Line vbuffer

      let mut line_vbuffer_ptrs = [std::ptr::null_mut::<LineVertex>();FRAMES_IN_FLIGHT];
      let mut line_vbuffers = std::array::from_fn(|_|make_buffer(&device, LINE_VBUFFER_CAP * std::mem::size_of::<LineVertex>(), D3D12_HEAP_TYPE_UPLOAD));

      for i in 0..FRAMES_IN_FLIGHT {
        let mut ptr = std::ptr::null_mut();
        line_vbuffers[i].Map(0, None, Some(&mut ptr));
        line_vbuffer_ptrs[i] = ptr as _;
      }

      let mut renderer = Renderer {
        dsv: dsv_heap.alloc(),
        camera_cbuffer: BufferedBuffer::new(&device, 1),
        frame: 0,
        device,
        queue,
        fence_val: 0,
        fence,
        frame_fences: [0;FRAMES_IN_FLIGHT],
        cmd_allocators,
        cmd_lists,
        swapchain_fences: [0;DXGI_MAX_SWAP_CHAIN_BUFFERS as _],
        swapchain_rtvs: std::array::from_fn(|i|rtv_heap.cpu_handle(rtvs[i])),
        swapchain_buffers: Vec::new(),
        swapchain_w: swapchain_desc.BufferDesc.Width,
        swapchain_h: swapchain_desc.BufferDesc.Height,
        swapchain,
        window,
        rtv_heap,
        cbv_srv_uav_heap,
        dsv_heap,

        line_pipeline,
        scene_pipeline,


        depth_buffer: None,
        static_mesh_pool: HandledPool::new(),
        transform_pool: HandledPool::new(),
        free_transforms: Vec::new(),

        line_vbuffer_ptrs,
        line_vbuffers,

        resident_resources
      };

      renderer.init_swapchain_resources();

      return Ok(renderer);
    }
  }

  pub fn acquire_frame(&self) {
    self.wait(self.frame_fences[self.frame]);
  }

  pub fn render(&mut self, meshes: &Vec<(StaticMesh, Transform)>, view_matrix: Matrix4<f32>) {
    unsafe{
      self.match_window_size();

      let swapchain_index = self.swapchain.GetCurrentBackBufferIndex() as usize;

      let cmd_list = &self.cmd_lists[self.frame];
      let cmd_allocator = &self.cmd_allocators[self.frame];

      cmd_allocator.Reset().unwrap();
      cmd_list.Reset(cmd_allocator, None).unwrap();

      // Start recording commands

      cmd_list.ResourceBarrier(&[transition_barrier(&self.swapchain_buffers[swapchain_index], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);

      let clear_rect = RECT {
        right: self.swapchain_w as i32,
        bottom: self.swapchain_h as i32,
        ..Default::default()
      };

      cmd_list.ClearDepthStencilView(self.dsv_heap.cpu_handle(self.dsv), D3D12_CLEAR_FLAG_DEPTH, 0.0, 0, &[clear_rect]);
      cmd_list.ClearRenderTargetView(self.swapchain_rtvs[swapchain_index], &[0.1, 0.1, 0.1, 1.0], None); 
      cmd_list.OMSetRenderTargets(1, Some(&self.swapchain_rtvs[swapchain_index]), None, Some(&self.dsv_heap.cpu_handle(self.dsv)));

      cmd_list.RSSetViewports(&[
        D3D12_VIEWPORT {
          Width: self.swapchain_w as f32,
          Height: self.swapchain_h as f32,
          MaxDepth: 1.0,
          ..Default::default()
        }
      ]);

      cmd_list.RSSetScissorRects(&[
        RECT {
          right: self.swapchain_w as _,
          bottom: self.swapchain_h as _,
          ..Default::default()
        }
      ]);




      // Render the scene

      self.scene_pipeline.bind(&cmd_list, &self.cbv_srv_uav_heap);

      let aspect = self.swapchain_w as f32 / self.swapchain_h as f32;
      let proj_matrix = perspective_inf_rev_z(aspect, 3.1415 * 0.5, 0.01);

      let view_proj = proj_matrix * view_matrix;
      self.camera_cbuffer.write(self.frame, 0, &view_proj);

      cmd_list.SetGraphicsRootConstantBufferView(0, self.camera_cbuffer.gpu_virtual_address(self.frame, 0));

      cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

      for (mesh_handle, transform) in meshes {
        let m = self.static_mesh_pool.get(*mesh_handle);
        let t = self.transform_pool.get(*transform);

        self.scene_pipeline.set_32bit_constant(&cmd_list, self.cbv_srv_uav_heap.verify_handle(m.vbuffer_srv) as u32, 0);
        self.scene_pipeline.set_32bit_constant(&cmd_list, self.cbv_srv_uav_heap.verify_handle(m.ibuffer_srv) as u32, 1);
        self.scene_pipeline.set_32bit_constant(&cmd_list, self.cbv_srv_uav_heap.verify_handle(t.cbvs[self.frame]) as u32, 2);

        cmd_list.DrawInstanced(m.index_count as u32, 1, 0, 0);
      }



      
      // Render some lines

      let test_cam= Matrix4::new_translation(&Vector3::new(-5.0, 2.0, 2.0));
      let test_view = test_cam.try_inverse().unwrap();
      let test_proj = perspective_inf_rev_z(1.0, std::f32::consts::PI * 0.5, 0.1);

      let test_inv_view_proj = (test_proj * test_view).try_inverse().unwrap();

      let box_points: Vec<Vector3<f32>> = [
        Vector3::new(-1.0,  1.0, 1.0),
        Vector3::new( 1.0,  1.0, 1.0),
        Vector3::new( 1.0, -1.0, 1.0),
        Vector3::new(-1.0, -1.0, 1.0),
        Vector3::new(-1.0,  1.0, 0.01),
        Vector3::new( 1.0,  1.0, 0.01),
        Vector3::new( 1.0, -1.0, 0.01),
        Vector3::new(-1.0, -1.0, 0.01),
      ].iter().map(|p|{
        let v = Vector4::new(p.x, p.y, p.z, 1.0);
        let h =  test_inv_view_proj * v;
        h.xyz() / h.w
      }).collect();

      let mut line_vertex_count: usize = 0;

      let mut push_line = |a: Vector3<f32>, b: Vector3<f32>| {
        for v in [a, b] {
          if line_vertex_count < LINE_VBUFFER_CAP {
            std::ptr::copy_nonoverlapping(&LineVertex{pos:v,pad:0.0}, self.line_vbuffer_ptrs[self.frame].add(line_vertex_count), 1);
            line_vertex_count += 1;
          }
        }
      };

      for i in 0..4 {
        push_line(box_points[i], box_points[(i+1)%4]);
        push_line(box_points[i], box_points[i+4]);
        push_line(box_points[i+4], box_points[(i+1)%4+4]);
      }

      self.line_pipeline.bind(&cmd_list, &self.cbv_srv_uav_heap);

      cmd_list.SetGraphicsRootConstantBufferView(0, self.camera_cbuffer.gpu_virtual_address(self.frame, 0));
      cmd_list.SetGraphicsRootShaderResourceView(1, self.line_vbuffers[self.frame].GetGPUVirtualAddress());

      cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_LINELIST);
      cmd_list.DrawInstanced(line_vertex_count as u32, 1, 0, 0);




      cmd_list.ResourceBarrier(&[transition_barrier(&self.swapchain_buffers[swapchain_index], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);

      // Submit command buffer
      cmd_list.Close().unwrap();
      let submit: [Option<ID3D12CommandList>;1] = [ Some(cmd_list.cast::<ID3D12CommandList>().unwrap()) ];

      self.queue.Wait(&self.fence, self.swapchain_fences[swapchain_index]).unwrap();
      self.queue.ExecuteCommandLists(&submit);
      self.frame_fences[self.frame] = self.signal();

      if self.swapchain.Present(1, DXGI_PRESENT::default()).is_err() {
        panic!("Presentation failed");
      }
      self.swapchain_fences[swapchain_index] = self.signal();

      self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
    };
  }

  fn match_window_size(&mut self) {
    let (ww, wh) = hwnd_size(self.window);

    if ww == 0 || wh == 0 {
      return;
    }

    if ww != self.swapchain_w || wh != self.swapchain_h {
      self.wait_device_idle();

      self.swapchain_buffers.clear();
      unsafe{self.swapchain.ResizeBuffers(0, ww, wh, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG::default()).unwrap()};

      self.init_swapchain_resources();

      self.swapchain_w = ww;
      self.swapchain_h = wh;
    }
  }

  pub fn new_static_mesh(&mut self, cpu_mesh: &CPUStaticMesh) -> StaticMesh {
      let vbuffer = make_buffer(&self.device, cpu_mesh.vertex_data.len() * std::mem::size_of::<Vertex>(), D3D12_HEAP_TYPE_UPLOAD);
      let ibuffer = make_buffer(&self.device, cpu_mesh.index_data.len() * std::mem::size_of::<u32>(), D3D12_HEAP_TYPE_UPLOAD);

      unsafe {
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();

        vbuffer.Map(0, None, Some(&mut ptr)).expect("Failed to map buffer");
        std::ptr::copy_nonoverlapping(cpu_mesh.vertex_data.as_ptr(), ptr as _, cpu_mesh.vertex_data.len());
        vbuffer.Unmap(0, None);

        ibuffer.Map(0, None, Some(&mut ptr)).expect("Failed to map buffer");
        std::ptr::copy_nonoverlapping(cpu_mesh.index_data.as_ptr(), ptr as _, cpu_mesh.index_data.len());
        ibuffer.Unmap(0, None);
      }

      let vbuffer_srv = self.cbv_srv_uav_heap.alloc();
      let ibuffer_srv = self.cbv_srv_uav_heap.alloc();

      unsafe {
        self.device.CreateShaderResourceView(&vbuffer, Some(&structured_buffer_srv_desc::<Vertex>(cpu_mesh.vertex_data.len())), self.cbv_srv_uav_heap.cpu_handle(vbuffer_srv));
        self.device.CreateShaderResourceView(&ibuffer, Some(&structured_buffer_srv_desc::<u32>(cpu_mesh.index_data.len())), self.cbv_srv_uav_heap.cpu_handle(ibuffer_srv));
      }

      let data = StaticMeshData {
        vbuffer,
        ibuffer,
        vbuffer_srv,
        ibuffer_srv,
        index_count: cpu_mesh.index_data.len()
      };

      return self.static_mesh_pool.alloc(data);
  }

  pub fn new_transform(&mut self) -> Transform {
    if self.free_transforms.is_empty() {
      const POOL_COUNT: usize = 16;

      let padded_size = pad_256(std::mem::size_of::<Matrix4<f32>>());

      let buffer = make_buffer(&self.device, POOL_COUNT * padded_size * FRAMES_IN_FLIGHT, D3D12_HEAP_TYPE_UPLOAD);

      let mut ptr = std::ptr::null_mut();
      unsafe{buffer.Map(0, None, Some(&mut ptr)).unwrap()};

      let base_virtual = unsafe{buffer.GetGPUVirtualAddress()};

      for i in 0..POOL_COUNT {
        let offset = |f: usize| -> usize {
          (i * FRAMES_IN_FLIGHT + f) * padded_size
        };

        let data = TransformData  {
          ptrs: std::array::from_fn(|f|unsafe{ptr.byte_add(offset(f))} as *mut Matrix4<f32>),
          cbvs: std::array::from_fn(|_|self.cbv_srv_uav_heap.alloc())
        };

        for (f, cbv) in data.cbvs.iter().enumerate() {
          unsafe {
            self.device.CreateConstantBufferView(
              Some(&cbv_desc(base_virtual + offset(f) as u64, padded_size)),
              self.cbv_srv_uav_heap.cpu_handle(*cbv)
            );
          }
        }

        for p in data.ptrs {
          unsafe{std::ptr::copy_nonoverlapping(&Matrix4::<f32>::identity(), p, 1)};
        }

        self.free_transforms.push(self.transform_pool.alloc(data));
      }

      self.resident_resources.push(buffer);
    }

    return self.free_transforms.pop().unwrap();
  }

  pub fn write_transform(&mut self, transform: Transform, matrix: &Matrix4::<f32>) {
    let data = self.transform_pool.get(transform);
    unsafe{std::ptr::copy_nonoverlapping(matrix, data.ptrs[self.frame], 1)};
  }

  fn init_swapchain_resources(&mut self) {
    unsafe {
      self.swapchain_buffers.clear();
      let desc = self.swapchain.GetDesc().unwrap();

      for i in 0..desc.BufferCount {
        self.swapchain_buffers.push(
          self.swapchain.GetBuffer(i).unwrap()
        );

        let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
          ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
          Format: desc.BufferDesc.Format,
          ..Default::default()
        };

        self.device.CreateRenderTargetView(&self.swapchain_buffers[i as usize], Some(&rtv_desc), self.swapchain_rtvs[i as usize]);
      }

      let depth_buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: self.swapchain_w as u64,
        Height: self.swapchain_h,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
          Count: 1,
          Quality: 0
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
      };

      let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
          Texture2D: D3D12_TEX2D_DSV {
            MipSlice: 0
          }
        },
        ..Default::default()
      };

      let clear_value = D3D12_CLEAR_VALUE {
        Format: dsv_desc.Format,
        Anonymous: D3D12_CLEAR_VALUE_0 {
          DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
            Depth: 0.0,
            Stencil: 0
          }
        }
      };

      let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
      };

      self.depth_buffer = None;
      self.device.CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &depth_buffer_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, Some(&clear_value), &mut self.depth_buffer).unwrap();

      self.device.CreateDepthStencilView(self.depth_buffer.as_ref().unwrap(), Some(&dsv_desc), self.dsv_heap.cpu_handle(self.dsv));
    }
  }

  fn signal(&mut self) -> u64 {
    self.fence_val += 1;
    unsafe{self.queue.Signal(&self.fence, self.fence_val).unwrap()};
    return self.fence_val;
  }

  fn wait(&self, val: u64) {
    unsafe{ 
      if self.fence.GetCompletedValue() < val {
        self.fence.SetEventOnCompletion(val, None).unwrap();
      }
    }
  }

  fn wait_device_idle(&mut self) {
    let val = self.signal();
    self.wait(val);
  }

}

struct PipelineCreateInfo<'a> {
  vs_code: ID3DBlob,
  ps_code: ID3DBlob,
  rtv_formats: &'a [DXGI_FORMAT],
  root_params: &'a [D3D12_ROOT_PARAMETER],
  primitive_topology_type: D3D12_PRIMITIVE_TOPOLOGY_TYPE
}

struct Pipeline {
  root_signature: ID3D12RootSignature,
  pipeline: ID3D12PipelineState,
  bindless_table_index: Option<u32>,
  constants_index: Option<u32>,
}

fn root_param_cbv(shader_visibility: D3D12_SHADER_VISIBILITY, register: u32, space: u32) -> D3D12_ROOT_PARAMETER {
  D3D12_ROOT_PARAMETER {
    ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
    ShaderVisibility: shader_visibility,
    Anonymous: D3D12_ROOT_PARAMETER_0 {
      Descriptor:  D3D12_ROOT_DESCRIPTOR {
        ShaderRegister: register,
        RegisterSpace: space 
      }
    }
  }
}

fn root_param_srv(shader_visibility: D3D12_SHADER_VISIBILITY, register: u32, space: u32) -> D3D12_ROOT_PARAMETER {
  D3D12_ROOT_PARAMETER {
    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
    ShaderVisibility: shader_visibility,
    Anonymous: D3D12_ROOT_PARAMETER_0 {
      Descriptor:  D3D12_ROOT_DESCRIPTOR {
        ShaderRegister: register,
        RegisterSpace: space 
      }
    }
  }
}

fn shader_input_type_to_descriptor_range_type(input: D3D_SHADER_INPUT_TYPE) -> D3D12_DESCRIPTOR_RANGE_TYPE {
  return match input {
    D3D_SIT_CBUFFER => D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
    D3D_SIT_TBUFFER => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    D3D_SIT_TEXTURE => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    D3D_SIT_SAMPLER => D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
    D3D_SIT_UAV_RWTYPED => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D_SIT_STRUCTURED => D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 
    D3D_SIT_UAV_RWSTRUCTURED => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D_SIT_BYTEADDRESS => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    D3D_SIT_UAV_RWBYTEADDRESS => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D_SIT_UAV_APPEND_STRUCTURED => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D_SIT_UAV_CONSUME_STRUCTURED => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    D3D_SIT_RTACCELERATIONSTRUCTURE => D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
    D3D_SIT_UAV_FEEDBACKTEXTURE => D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
    _ => panic!("invalid shader input type given")
  }
}

impl Pipeline {
  fn create(device: &ID3D12Device, info: &PipelineCreateInfo) -> std::result::Result<Pipeline, &'static str> {
    let mut root_params = info.root_params.to_vec();
    let mut ranges = Vec::<D3D12_DESCRIPTOR_RANGE>::new();

    let mut bindless_table_index = None;
    let mut constants_index = None;

    unsafe {
      let container: IDxcContainerReflection = DxcCreateInstance(&CLSID_DxcContainerReflection).map_err(|_|"failed to reflect")?;
      let blob = info.vs_code.clone().cast::<IDxcBlob>().unwrap();
      container.Load(&blob);

      let kind = u32::from_le_bytes([b'D', b'X', b'I', b'L']);
      let part_index = container.FindFirstPartKind(kind).unwrap();

      let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
      container.GetPartReflection(part_index, &ID3D12ShaderReflection::IID, &mut ptr).map_err(|_|"failed to reflect")?;

      let reflection = ID3D12ShaderReflection::from_raw(ptr);

      let mut desc = D3D12_SHADER_DESC::default();
      reflection.GetDesc(&mut desc);

      for i in 0..desc.BoundResources {
        let mut binding = D3D12_SHADER_INPUT_BIND_DESC::default();
        reflection.GetResourceBindingDesc(i, &mut binding);

        let name = std::ffi::CStr::from_ptr(binding.Name.0 as _).to_str().unwrap();

        if name == "Constants" {
          constants_index = Some(root_params.len() as u32);
          
          let cbuffer = reflection.GetConstantBufferByName(windows::core::s!("Constants")).unwrap();
          let mut cbuffer_desc = D3D12_SHADER_BUFFER_DESC::default();
          cbuffer.GetDesc(&mut cbuffer_desc);

          root_params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
              Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: binding.BindPoint,
                RegisterSpace: binding.Space,
                Num32BitValues: cbuffer_desc.Size / 4
              }
            }
          });
        }
        else if binding.BindCount == 0xffffffff || binding.BindCount == 0 {
          // Bindless table
          ranges.push(D3D12_DESCRIPTOR_RANGE{
            RangeType: shader_input_type_to_descriptor_range_type(binding.Type),
            NumDescriptors: u32::MAX,
            BaseShaderRegister: binding.BindPoint,
            RegisterSpace: binding.Space,
            OffsetInDescriptorsFromTableStart: 0
          });
        }
      }
    }

    if !ranges.is_empty() {
      bindless_table_index = Some(root_params.len() as u32);

      root_params.push(D3D12_ROOT_PARAMETER{
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
          DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
            NumDescriptorRanges: ranges.len() as u32,
            pDescriptorRanges: ranges.as_ptr()
          }
        },
      });
    }

    let root_signature_desc = D3D12_ROOT_SIGNATURE_DESC {
      pParameters: root_params.as_ptr(),
      NumParameters: root_params.len() as u32,
      ..Default::default()
    };

    let mut root_signature_code_opt: Option<ID3DBlob> = None;

    unsafe{
      D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut root_signature_code_opt, None).map_err(|_|"Failed to serialize root signature")?;
    }

    let root_signature_code = root_signature_code_opt.unwrap();

    let root_signature= unsafe {device.CreateRootSignature(
      0,
      std::slice::from_raw_parts(root_signature_code.GetBufferPointer() as _, root_signature_code.GetBufferSize())
    ).map_err(|_|"Failed to create root signature")?};

    let mut pipeline_desc = unsafe { D3D12_GRAPHICS_PIPELINE_STATE_DESC {
      pRootSignature: unsafe{std::mem::transmute_copy(&root_signature)},

      VS: D3D12_SHADER_BYTECODE {
        pShaderBytecode: info.vs_code.GetBufferPointer(),
        BytecodeLength: info.vs_code.GetBufferSize()
      },

      PS: D3D12_SHADER_BYTECODE {
        pShaderBytecode: info.ps_code.GetBufferPointer() as _,
        BytecodeLength: info.ps_code.GetBufferSize()
      },

      BlendState: D3D12_BLEND_DESC {
        RenderTarget: std::array::from_fn(|_|D3D12_RENDER_TARGET_BLEND_DESC{
          SrcBlend:	D3D12_BLEND_ONE,
          BlendOp:	D3D12_BLEND_OP_ADD,
          SrcBlendAlpha:	D3D12_BLEND_ONE,
          BlendOpAlpha:	D3D12_BLEND_OP_ADD,
          LogicOp:	D3D12_LOGIC_OP_NOOP,
          RenderTargetWriteMask: 0b1111,
          ..Default::default()
        }),
        ..Default::default()
      },

      SampleMask: 0xffffffff,

      RasterizerState: D3D12_RASTERIZER_DESC {
        FillMode:	D3D12_FILL_MODE_SOLID,
        CullMode:	D3D12_CULL_MODE_BACK,
        FrontCounterClockwise: TRUE,
        DepthClipEnable:	TRUE,
        ..Default::default()
      },

      DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
        DepthEnable:	TRUE,
        DepthWriteMask:	D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc:	D3D12_COMPARISON_FUNC_GREATER,
        StencilEnable: FALSE,
        StencilReadMask:	D3D12_DEFAULT_STENCIL_READ_MASK as _,
        StencilWriteMask:	D3D12_DEFAULT_STENCIL_WRITE_MASK as _,

        BackFace: D3D12_DEPTH_STENCILOP_DESC{
          StencilFailOp: D3D12_STENCIL_OP_KEEP,
          StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
          StencilPassOp: D3D12_STENCIL_OP_KEEP,
          StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        }, 

        FrontFace: D3D12_DEPTH_STENCILOP_DESC{
          StencilFailOp: D3D12_STENCIL_OP_KEEP,
          StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
          StencilPassOp: D3D12_STENCIL_OP_KEEP,
          StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        }, 
      },

      PrimitiveTopologyType: info.primitive_topology_type,

      NumRenderTargets: info.rtv_formats.len() as u32,

      SampleDesc: DXGI_SAMPLE_DESC {
        Count: 1,
        Quality: 0
      },

      DSVFormat: DXGI_FORMAT_D32_FLOAT,
      
      ..Default::default()
    }};

    for (i, f) in info.rtv_formats.iter().enumerate() {
      pipeline_desc.RTVFormats[i] = *f;
    }

    let pipeline = unsafe{device.CreateGraphicsPipelineState(&pipeline_desc).map_err(|_|"Failed to create pipeline")?};

    return Ok(Pipeline {
      root_signature,
      pipeline,
      bindless_table_index,
      constants_index,
    });
  }

  fn bind(&self, cmd_list: &ID3D12GraphicsCommandList, bindless_heap: &DescriptorHeap) {
    unsafe {
      cmd_list.SetGraphicsRootSignature(&self.root_signature);
      cmd_list.SetPipelineState(&self.pipeline);

      cmd_list.SetDescriptorHeaps(&[Some(bindless_heap.heap.clone())]);

      if let Some(i) = self.bindless_table_index {
        cmd_list.SetGraphicsRootDescriptorTable(i, bindless_heap.gpu_base);
      }
    }
  }

  fn set_32bit_constant(&self, cmd_list: &ID3D12GraphicsCommandList, value: u32, index: u32) {
    unsafe {
      cmd_list.SetGraphicsRoot32BitConstant(self.constants_index.unwrap(), value, index);
    }
  }
}
  

fn transition_barrier(resource: &ID3D12Resource, state_before: D3D12_RESOURCE_STATES, state_after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
  return D3D12_RESOURCE_BARRIER {
    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,

    Anonymous: D3D12_RESOURCE_BARRIER_0 {
      Transition: unsafe{std::mem::transmute_copy(&D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: std::mem::transmute_copy(resource),
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: state_before,
        StateAfter: state_after,
      })}
    },
      
    ..Default::default()
  };
}

fn load_shader(path: &str) -> std::result::Result<ID3DBlob, String> {
  let mut file = std::fs::File::open(path).map_err(|_|format!("Missing file {}", path))?;
  let len = file.seek(SeekFrom::End(0)).unwrap() as usize;

  let blob = unsafe{ D3DCreateBlob(len) }.unwrap();
  let slice = unsafe{std::slice::from_raw_parts_mut(blob.GetBufferPointer() as *mut u8, blob.GetBufferSize())};

  file.seek(SeekFrom::Start(0));

  file.read(slice).map_err(|_|"failed to read shader code");

  return Ok(blob);
}

impl Drop for Renderer {
  fn drop(&mut self) {
    self.wait_device_idle();
  }
}

fn hwnd_size(window: HWND) -> (u32, u32) {
  let mut rect = RECT::default();
  unsafe{GetClientRect(window, &mut rect).unwrap()};
  return (
    (rect.right - rect.left) as _,
    (rect.bottom - rect.top) as _
  );
}

#[allow(dead_code)]
struct DescriptorHeap {
  id: u8,
  heap: ID3D12DescriptorHeap, // Need to keep reference
  free_list: Vec<u32>,
  generation: Vec<u32>,
  stride: usize,
  cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
  gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE
}

type DescriptorHandle = u64;

static NEXT_DESCRIPTOR_ID: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);

#[allow(dead_code)]
impl DescriptorHeap {
  fn new(device: &ID3D12Device, capacity: u32, desc_type: D3D12_DESCRIPTOR_HEAP_TYPE, flags: D3D12_DESCRIPTOR_HEAP_FLAGS) -> std::result::Result<DescriptorHeap, &'static str> {
    unsafe{
      let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: desc_type,
        NumDescriptors: capacity,
        Flags: flags,
        ..Default::default()
      };

      let heap: ID3D12DescriptorHeap = match device.CreateDescriptorHeap(&heap_desc) {
        Ok(heap) => {heap},
        Err(_) => { return Err("Failed to create descriptor heap"); }
      };

      return Ok(DescriptorHeap {
        id: NEXT_DESCRIPTOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        cpu_base: heap.GetCPUDescriptorHandleForHeapStart(),

        gpu_base: if (flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) != D3D12_DESCRIPTOR_HEAP_FLAG_NONE {
          heap.GetGPUDescriptorHandleForHeapStart()
        }
        else {
          D3D12_GPU_DESCRIPTOR_HANDLE{..Default::default()}
        },

        heap,
        free_list: (0..capacity).rev().collect(),
        generation: vec![1;capacity as usize],
        stride: device.GetDescriptorHandleIncrementSize(desc_type) as usize,
      });
    }
  }

  fn alloc(&mut self) -> DescriptorHandle {
    let index = self.free_list.pop().expect("Descriptor heap exhausted");
    let gen = self.generation[index as usize];
    return (((gen | ((self.id as u32) << 24)) as u64) << 32) | index as u64;
  }

  fn unsafe_handle_index(handle: DescriptorHandle) -> usize {
    return (handle & 0xffffffff) as usize;
  }

  fn handle_metadata(handle: DescriptorHandle) -> (u32, u8) {
    let shifted = (handle >> 32) as u32;
    return (shifted & 0xffffff, (shifted >> 24) as u8);
  }

  fn verify_handle(&self, handle: DescriptorHandle) -> usize {
    let index = Self::unsafe_handle_index(handle);
    let (gen, id) = Self::handle_metadata(handle);

    std::assert!(index < self.generation.len());
    std::assert!(id == self.id);
    std::assert!(gen == self.generation[index]);

    return index;
  }

  fn free(&mut self, handle: DescriptorHandle) {
    let index = self.verify_handle(handle);
    self.generation[index] += 1; // Kill the handle
    self.free_list.push(index as u32);
  }

  fn cpu_handle(&self, handle: DescriptorHandle) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    let index = self.verify_handle(handle);
    return D3D12_CPU_DESCRIPTOR_HANDLE {
      ptr: self.cpu_base.ptr + index * self.stride
    };
  }

  fn gpu_handle(&self, handle: DescriptorHandle) -> D3D12_GPU_DESCRIPTOR_HANDLE {
    let index = self.verify_handle(handle);
    return D3D12_GPU_DESCRIPTOR_HANDLE {
      ptr: self.gpu_base.ptr + (index * self.stride) as u64
    };
  }
}

fn make_buffer(device: &ID3D12Device, size: usize, heap_type: D3D12_HEAP_TYPE) -> ID3D12Resource {
  unsafe {
    let desc = D3D12_RESOURCE_DESC {
      Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
      Width: size as u64,
      Height: 1,
      DepthOrArraySize: 1,
      MipLevels: 1,
      SampleDesc: DXGI_SAMPLE_DESC {
        Count: 1,
        Quality: 0
      },
      Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      ..Default::default()
    };

    let heap_props = D3D12_HEAP_PROPERTIES {
      Type: heap_type,
      ..Default::default()
    };

    let mut resource_opt: Option<ID3D12Resource> = None;
    device.CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource_opt).expect("Buffer creation failed");

    return resource_opt.unwrap();
  }
}

#[allow(unused)]
struct BufferedBuffer<T> {
  resource: ID3D12Resource,
  ptrs: [Vec<*mut T>; FRAMES_IN_FLIGHT],
  base_virtual_addr: u64,
  count: usize
}

impl<T> BufferedBuffer<T> {
  fn padded_size() -> usize {
    let sz = std::mem::size_of::<T>();
    return (sz + 255) & !(255 as usize);
  }

  fn new(device: &ID3D12Device, count: usize) -> Self {
    unsafe {
      let resource = make_buffer(device, Self::padded_size() * count * FRAMES_IN_FLIGHT, D3D12_HEAP_TYPE_UPLOAD);

      let mut void_ptr = std::ptr::null_mut::<std::ffi::c_void>();
      resource.Map(0, None, Some(&mut void_ptr)).expect("Failed to map buffer");
      let ptr_base = void_ptr as *mut T;

      let ptrs = std::array::from_fn(|frame|{
        (0..count).map(|index|ptr_base.byte_add((frame*count+index) * Self::padded_size())).collect()
      });

      return Self {
        base_virtual_addr: resource.GetGPUVirtualAddress(),
        resource,
        ptrs,
        count
      };
    }
  }

  fn write(&self, frame: usize, index: usize, data: &T) {
    unsafe{std::ptr::copy_nonoverlapping(data as _, self.ptrs[frame][index], 1)};
  }

  fn gpu_virtual_address(&self, frame: usize, index: usize) -> u64 {
    return self.base_virtual_addr + ((frame*self.count+index) * Self::padded_size()) as u64;
  }
}

fn structured_buffer_srv_desc<T>(num_elements: usize) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
  return D3D12_SHADER_RESOURCE_VIEW_DESC {
    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
      Buffer: D3D12_BUFFER_SRV {
        FirstElement: 0,
        NumElements: num_elements as u32,
        StructureByteStride: std::mem::size_of::<T>() as u32,
        ..Default::default()
      }
    },
    ..Default::default()
  };
}

fn cbv_desc(location: u64, padded_size: usize) -> D3D12_CONSTANT_BUFFER_VIEW_DESC {
  return D3D12_CONSTANT_BUFFER_VIEW_DESC {
    BufferLocation: location,
    SizeInBytes: padded_size as u32
  };
}

#[allow(unused)]
struct StaticMeshData {
  vbuffer: ID3D12Resource,
  ibuffer: ID3D12Resource,

  vbuffer_srv: DescriptorHandle,
  ibuffer_srv: DescriptorHandle,

  index_count: usize,
}

pub struct CPUStaticMesh {
  pub vertex_data: Vec<Vertex>,
  pub index_data: Vec<u32>
}

trait HandledPoolHandle {
  fn generation(&self) -> u32;
  fn index(&self) -> usize;
  fn make(index: u32, generation: u32) -> Self;
}

struct HandledPool<T, H: HandledPoolHandle> {
  data: Vec<Option<T>>,
  generation: Vec<u32>,
  free_list: Vec<usize>,
  phantom: std::marker::PhantomData<H>
}

impl<T, H: HandledPoolHandle> HandledPool<T, H> {
  fn new() -> Self {
    return HandledPool {
      data: Vec::new(),
      generation: Vec::new(),
      free_list: Vec::new(),
      phantom: std::marker::PhantomData::default()
    }
  }

  fn alloc(&mut self, data: T) -> H {
    if self.free_list.is_empty() {
      self.free_list.push(self.data.len());
      self.data.push(None);
      self.generation.push(1);
    }

    let index = self.free_list.pop().unwrap();
    let generation = self.generation[index];
    self.data[index] = Some(data);

    return H::make(index as u32, generation);
  }

  fn verify_handle(&self, handle: H) -> usize {
    std::assert!(handle.index() < self.data.len());
    std::assert!(handle.generation() == self.generation[handle.index()]);
    return handle.index();
  }

  fn free(&mut self, handle: H) {
    let idx = self.verify_handle(handle);
    self.generation[idx] += 1;
    self.data[idx] = None;
    self.free_list.push(idx);
  }

  fn get(&mut self, handle: H) -> &mut T {
    let idx = self.verify_handle(handle);
    return self.data[idx].as_mut().unwrap();
  }
}

struct TransformData {
  cbvs: [DescriptorHandle; FRAMES_IN_FLIGHT],
  ptrs: [*mut Matrix4<f32>; FRAMES_IN_FLIGHT]
}

#[repr(packed)]
struct LineVertex {
  pos: Vector3<f32>,
  pad: f32
}

fn perspective_inf_rev_z(aspect: f32, fovy: f32, znear: f32) -> Matrix4<f32> {
  let mut res = Matrix4::<f32>::zeros();

  let tan_fov = (fovy*0.5).tan();

  res[(0, 0)] = 1.0 / (tan_fov * aspect);
  res[(1, 1)] = 1.0 / tan_fov;

  res[(2, 2)] = 0.0;
  res[(2, 3)] = znear;

  res[(3, 2)] = -1.0;

  return res;
}