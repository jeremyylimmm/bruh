use std::result::Result;
use std::collections::HashMap;
use crate::*;

#[derive(Debug)]
enum ComponentType {
  Invalid,
  F32,
  U32,
  U16
}

#[derive(Debug)]
struct BufferView {
  buffer: usize,
  length: usize,
  offset: usize,
}

#[derive(Debug)]
struct Accessor {
  buffer_view: usize,
  component_type: ComponentType,
  count: usize,
  dims: usize,
  offset: usize
}

#[derive(Debug)]
struct Primitive {
  attributes: HashMap<String, usize>,
  indices: usize
}

#[derive(Debug)]
struct Mesh<'a> {
  name: &'a String,
  primitives: Vec<Primitive>
}

#[derive(Debug)]
struct Node<'a> {
  mesh: Option<usize>,
  name: &'a String
}

#[derive(Debug)]
struct Scene<'a> {
  name: &'a String,
  nodes: Vec<usize>
}

impl ComponentType {
  fn new(code: i64) -> Self {
    return match code {
      0x1406 => Self::F32,
      0x1403 => Self::U16,
      0x1405 => Self::U32,
      _ => Self::Invalid,
    };
  }
}

fn type_to_dims(str: &String) -> Option<usize> {
  if str == "SCALAR" {
    return Some(1);
  }

  let rest =  str.trim_start_matches("VEC");
  return rest.parse().ok();
}

pub fn load(path: &str) -> Result<Vec<renderer::CPUStaticMesh>, &'static str> {
  let dir= match std::path::Path::new(path).parent() {
    Some(p) => {
      p.to_str().ok_or("no parent directory")?
    },
    _ => {
      ""
    }
  };

  let gltf_text = std::fs::read_to_string(path).map_err(|_|"failed to load gltf file")?; 

  // Parse the JSON tree

  let root_n = json::parse(&gltf_text).map_err(|_|"gltf json could not be parsed")?;
  let root = root_n.as_obj().ok_or("root not object")?;

  // Verify the version

  let asset = root.get("asset").ok_or("no asset")?.as_obj().ok_or("asset not object")?;
  let version = asset.get("version").ok_or("no version")?.as_string().ok_or("version not string")?;

  if version != "2.0" {
    return Err("only gltf 2.0 assets supported");
  }

  // Load all buffers

  let mut buffers = Vec::<Vec<u8>>::new();
  
  for b in root.get("buffers").ok_or("no buffers")?.as_array().ok_or("buffers not array")? {
    let buf = b.as_obj().ok_or("buffer not object")?;

    let _length = buf.get("byteLength").ok_or("no buffer byteLength")?.as_integer().ok_or("buffer byteLength not integer")?;
    let uri = buf.get("uri").ok_or("no buffer uri")?.as_string().ok_or("buffer uri not string")?;

    let buf_path = format!("{}/{}", dir, uri);

    let data = std::fs::read(buf_path).map_err(|_|"failed to load buffer")?;
    buffers.push(data);
  }

  // Gather buffer views

  let mut buffer_views = Vec::<BufferView>::new();

  for bv in root.get("bufferViews").ok_or("no buffer views")?.as_array().ok_or("buffer views not array")? {
    let buf_view = bv.as_obj().ok_or("buffer view not object")?;

    let buf = buf_view.get("buffer").ok_or("no buffer")?.as_integer().ok_or("buffer view buffer not integer")?;
    let length = buf_view.get("byteLength").ok_or("no buffer view length")?.as_integer().ok_or("buffer view length not integer")?;
    let offset = buf_view.get("byteOffset").ok_or("no buffer view offset")?.as_integer().ok_or("buffer view offset not integer")?;

    buffer_views.push(BufferView{
      buffer: buf as usize,
      length: length as usize,
      offset: offset as usize
    });
  }

  // Gather accessors

  let mut accessors = Vec::<Accessor>::new();

  for a in root.get("accessors").ok_or("no accessors")?.as_array().ok_or("accessors not array")? {
    let acc = a.as_obj().ok_or("accessor not object")?;
    
    let buf_view = acc.get("bufferView").ok_or("no accessor buffer view")?.as_integer().ok_or("accessor buffer view not integer")?;
    let comp_type = acc.get("componentType").ok_or("no accessor component type")?.as_integer().ok_or("accessor component type not integer")?;
    let count = acc.get("count").ok_or("no accessor count")?.as_integer().ok_or("accessor count not integer")?;
    let ty = acc.get("type").ok_or("no accessor type")?.as_string().ok_or("accessor type not string")?;

    let offset = match acc.get("byteOffset") {
      Some(off) => off.as_integer().ok_or("accessor offset not integer")?,
      None => 0
    };

    accessors.push(Accessor{
      buffer_view: buf_view as usize,
      component_type: ComponentType::new(comp_type),
      count: count as usize,
      dims: type_to_dims(ty).ok_or("invalid accessor type")?,
      offset: offset as usize
    });
  }

  // Gather meshes

  let mut meshes = Vec::<Mesh>::new();

  for m in root.get("meshes").ok_or("no meshes")?.as_array().ok_or("meshes not array")? {
    let msh = m.as_obj().ok_or("mesh not object")?;

    let name = msh.get("name").ok_or("no mesh name")?.as_string().ok_or("mesh name not string")?;
    let mut prims = Vec::<Primitive>::new();

    for p in msh.get("primitives").ok_or("no primitives")?.as_array().ok_or("mesh primitives not array")? {
      let prim = p.as_obj().ok_or("primitive not object")?;
      let attribs = prim.get("attributes").ok_or("no primitive attributes")?;

      let mut attributes = HashMap::<String, usize>::new();

      for (name, n) in attribs.as_obj().ok_or("primitive attributes not object")? {
        let val = n.as_integer().ok_or("primitive attribute not integer")?;
        attributes.insert(name.clone(), val as usize);
      }

      let indices = prim.get("indices").ok_or("no primitive indices")?.as_integer().ok_or("primitive indices not integer")?;

      prims.push(Primitive{
        attributes: attributes,
        indices: indices as usize
      });
    }

    meshes.push(Mesh{
      name,
      primitives: prims
    });
  }

  // Gather nodes

  let mut nodes = Vec::<Node>::new();

  for n in root.get("nodes").ok_or("no nodes")?.as_array().ok_or("nodes not array")? {
    let node = n.as_obj().ok_or("node not object")?;

    let mesh = match node.get("mesh") {
      Some(m) => Some(m.as_integer().ok_or("node mesh not integer")? as usize),
      None => None
    };

    let name = node.get("name").ok_or("no node name")?.as_string().ok_or("node name node string")?;

    nodes.push(Node{
      mesh,
      name
    });
  }

  // Gather scenes

  let mut scenes = Vec::<Scene>::new();

  for s in root.get("scenes").ok_or("no scenes")?.as_array().ok_or("scenes not array")? {
    let scene = s.as_obj().ok_or("scene not object")?;

    let name = scene.get("name").ok_or("no scene name")?.as_string().ok_or("scene name not string")?;
    let mut nds = Vec::<usize>::new();

    for n in scene.get("nodes").ok_or("no scene nodes")?.as_array().ok_or("scene nodes not array")? {
      nds.push(n.as_integer().ok_or("scene node not integer")? as usize);
    }

    scenes.push(Scene {
      name,
      nodes: nds
    });
  }

  // Root scene
  let _scene = root.get("scene").ok_or("no root scene")?.as_integer().ok_or("root scene not integer")? as usize;

  let mut static_meshes = Vec::<renderer::CPUStaticMesh>::new();

  // Process all meshes into static meshes
  for m in &meshes {
    for p in &m.primitives
    {
      let get_accessor = |name: &str| -> Result<&Accessor, &'static str> {
        let index = *p.attributes.get(name).ok_or("primitive missing attribute")?;
        return Ok(accessors.get(index).ok_or("primitive attribute out of bounds")?);
      };

      let check_f32 = |acc: &Accessor| -> Result<(), &str> {
        if let ComponentType::F32 = acc.component_type {
          return Ok(());
        }
        else {
          return Err("primitive's attribute was not f32 type");
        }
      };

      let check_dims = |acc: &Accessor, dims: usize| -> Result<(), &str> {
        if acc.dims == dims {
          return Ok(());
        }
        else {
          return Err("primitive's attribute had unexpected number of dimensions");
        }
      };

      let accessor_data = |acc: &Accessor| -> Result<*const std::ffi::c_void, &str> {
        let buffer_view = buffer_views.get(acc.buffer_view).ok_or("accessor buffer view out of bounds")?;
        let buffer = buffers.get(buffer_view.buffer).ok_or("buffer view buffer out of bounds")?;
        return Ok(unsafe{buffer.as_ptr().byte_add(buffer_view.offset + acc.offset)} as _);
      };

      let pos_acc  = get_accessor("POSITION")?;
      let norm_acc = get_accessor("NORMAL")?;
      let tex_acc  = get_accessor("TEXCOORD_0")?;
      let ind_acc = accessors.get(p.indices).ok_or("primitive indices accessor out of bounds")?;

      check_f32(pos_acc)?;
      check_f32(norm_acc)?;
      check_f32(tex_acc)?;

      check_dims(pos_acc, 3)?;
      check_dims(norm_acc, 3)?;
      check_dims(tex_acc, 2)?;

      check_dims(ind_acc, 1)?;

      if pos_acc.count != norm_acc.count || pos_acc.count != tex_acc.count {
        return Err("primitive's attributes have differing 'counts'");
      }

      let vertex_count = pos_acc.count;
      let index_count = ind_acc.count;

      let mut vertex_data = Vec::<renderer::Vertex>::with_capacity(vertex_count);
      let mut index_data = Vec::<u32>::with_capacity(index_count);

      let pos_src  = unsafe{std::slice::from_raw_parts(accessor_data(pos_acc)?  as *const f32, vertex_count * 3)};
      let norm_src = unsafe{std::slice::from_raw_parts(accessor_data(norm_acc)? as *const f32, vertex_count * 3)};
      let tex_src  = unsafe{std::slice::from_raw_parts(accessor_data(tex_acc)?  as *const f32, vertex_count * 2)};

      for i in 0..vertex_count {
        vertex_data.push(renderer::Vertex{
          pos: [
            pos_src[i*3 + 0],
            pos_src[i*3 + 1],
            pos_src[i*3 + 2],
          ],
          norm: [
            norm_src[i*3 + 0],
            norm_src[i*3 + 1],
            norm_src[i*3 + 2],
          ],
          tex_coord: [
            tex_src[i*2 + 0],
            tex_src[i*2 + 1],
          ],
        });
      }

      match ind_acc.component_type {
        ComponentType::U32 => {
          let ind_src = unsafe{std::slice::from_raw_parts(accessor_data(ind_acc)? as *const u32, index_count)};

          for i in 0..index_count {
            index_data.push(ind_src[i]);
          }
        }
        ComponentType::U16 => {
          let ind_src = unsafe{std::slice::from_raw_parts(accessor_data(ind_acc)? as *const u16, index_count)};

          for i in 0..index_count {
            index_data.push(ind_src[i] as u32);
          }
        }
        _ => {
          return Err("invalid primitive indices accessor type");
        }
      }

      static_meshes.push(renderer::CPUStaticMesh{
        vertex_data,
        index_data
      });
    }
  }

  return Ok(static_meshes);
}