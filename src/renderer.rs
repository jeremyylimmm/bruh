use windows::Win32::{Graphics::Direct3D12::*, Graphics::Direct3D::*, Graphics::Dxgi::Common::*, Graphics::Dxgi::*, Foundation::*, UI::WindowsAndMessaging::*};
use windows::core::*;

const FRAMES_IN_FLIGHT: usize = 2;

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
  root_signature: ID3D12RootSignature,
  pipeline: ID3D12PipelineState,
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

      let rtvs: [DescriptorHandle;DXGI_MAX_SWAP_CHAIN_BUFFERS as _] = std::array::from_fn(|_|rtv_heap.alloc());

      let vs_code = load_shader("shaders/triangle.vso")?;
      let ps_code = load_shader("shaders/triangle.pso")?;

      let root_signature_desc = D3D12_ROOT_SIGNATURE_DESC {
        ..Default::default()
      };

      let mut root_signature_code_opt: Option<ID3DBlob> = None;
      D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &mut root_signature_code_opt, None).map_err(|_|"Failed to serialize root signature")?;
      let root_signature_code = root_signature_code_opt.unwrap();

      let root_signature = device.CreateRootSignature(
        0,
        std::slice::from_raw_parts(root_signature_code.GetBufferPointer() as _, root_signature_code.GetBufferSize())
      ).map_err(|_|"Failed to create root signature")?;

      let mut pipeline_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: std::mem::transmute_copy(&root_signature),

        VS: D3D12_SHADER_BYTECODE {
          pShaderBytecode: vs_code.as_ptr() as _,
          BytecodeLength: vs_code.len()
        },

        PS: D3D12_SHADER_BYTECODE {
          pShaderBytecode: ps_code.as_ptr() as _,
          BytecodeLength: ps_code.len()
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
          DepthEnable:	FALSE,
          DepthWriteMask:	D3D12_DEPTH_WRITE_MASK_ALL,
          DepthFunc:	D3D12_COMPARISON_FUNC_LESS,
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

        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,

        NumRenderTargets: 1,

        SampleDesc: DXGI_SAMPLE_DESC {
          Count: 1,
          Quality: 0
        },
        
        ..Default::default()
      };

      pipeline_desc.RTVFormats[0] = swapchain_desc.BufferDesc.Format;

      let pipeline = device.CreateGraphicsPipelineState(&pipeline_desc).map_err(|_|"Failed t create pipeline")?;
      
      let mut renderer = Renderer {
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
        root_signature,
        pipeline,
      };

      renderer.init_swapchain_resources();

      return Ok(renderer);
    }
  }

  pub fn render(&mut self) {
    unsafe{
      self.match_window_size();

      let swapchain_index = self.swapchain.GetCurrentBackBufferIndex() as usize;

      let cmd_list = &self.cmd_lists[self.frame];
      let cmd_allocator = &self.cmd_allocators[self.frame];

      self.wait(self.frame_fences[self.frame]);

      cmd_allocator.Reset().unwrap();
      cmd_list.Reset(cmd_allocator, None).unwrap();

      // Start recording commands

      cmd_list.ResourceBarrier(&[transition_barrier(&self.swapchain_buffers[swapchain_index], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);

      cmd_list.ClearRenderTargetView(self.swapchain_rtvs[swapchain_index], &[0.2, 0.3, 0.3, 1.0], None); 
      cmd_list.OMSetRenderTargets(1, Some(&self.swapchain_rtvs[swapchain_index]), None, None);

      cmd_list.SetGraphicsRootSignature(&self.root_signature);
      cmd_list.SetPipelineState(&self.pipeline);

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

      cmd_list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      cmd_list.DrawInstanced(3, 1, 0, 0);

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

fn transition_barrier(resource: &ID3D12Resource, state_before: D3D12_RESOURCE_STATES, state_after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
  return D3D12_RESOURCE_BARRIER {
    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,

    Anonymous: D3D12_RESOURCE_BARRIER_0 {
      Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
        pResource: unsafe{std::mem::transmute_copy(resource)},
        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        StateBefore: state_before,
        StateAfter: state_after,
      })
    },
      
    ..Default::default()
  };
}

fn load_shader(path: &str) -> std::result::Result<Vec<u8>, String> {
  return std::fs::read(path).map_err(|_|format!("Missing file {}", path));
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

struct DescriptorHeap {
  id: u8,
  heap: ID3D12DescriptorHeap,
  free_list: Vec<u32>,
  generation: Vec<u32>,
  stride: usize,
  cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
  gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE
}

type DescriptorHandle = u64;

static NEXT_DESCRIPTOR_ID: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);

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
        free_list: (0..capacity).collect(),
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