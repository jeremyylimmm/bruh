use std::io::{Read, Seek};
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
  name: &'a str,
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

unsafe fn read_type<T>(buf: &mut std::io::BufReader<std::fs::File>) -> Option<T> {
  let mut x: T = std::mem::zeroed();
  let slice = std::slice::from_raw_parts_mut(&mut x as *mut T as *mut u8, std::mem::size_of::<T>());
  buf.read_exact(slice).ok()?;
  return Some(x);
}

#[repr(packed)]
struct GLBHeader {
  magic: u32,
  version: u32,
  length: u32
}

#[repr(packed)]
struct ChunkHeader {
  length: u32,
  ty: u32
}

pub struct SceneNode {
  pub name: String,
  pub local_transform: Matrix4<f32>,
  pub children: Vec<usize>,
  pub mesh: Option<usize>
}

pub fn load(path_str: &str) -> Result<Contents, &'static str> {
  let path = std::path::Path::new(path_str);

  let dir= match path.parent() {
    Some(p) => {
      p.to_str().ok_or("no parent directory")?
    },
    _ => {
      ""
    }
  };

  let ext = match path.extension() {
    Some(s) => {
      s
    }
    None => {
      return Err("cannot deduce file format without extension");
    }
  };

  let (json_text, buffers) = if ext == "gltf" {
    let gltf_text = std::fs::read_to_string(path).map_err(|_|"failed to load gltf file")?; 
    (gltf_text, None)
  }
  else {
    let file = std::fs::File::open(path).map_err(|_|"failed to load glb file")?;
    let mut buf = std::io::BufReader::new(file);

    let header = unsafe {read_type::<GLBHeader>(&mut buf).ok_or("failed to read glb header")?};

    if header.magic != 0x46546C67 {
      return Err("unrecognized format");
    }

    if header.version != 2 {
      return Err("unrecognized glb version");
    }

    const JSON_CHUNK: u32 = 0x4E4F534A;
    const BIN_CHUNK: u32 = 0x004E4942;

    let mut json_text = Option::<String>::None;
    let mut buffers = Vec::<Vec<u8>>::new();

    while buf.stream_position().unwrap() < header.length as u64 {
      let chunk_hdr = unsafe{read_type::<ChunkHeader>(&mut buf).ok_or("failed to read chunk header")?};

      let mut bytes = vec![0 as u8;chunk_hdr.length as usize];
      buf.read_exact(&mut bytes).map_err(|_|"failed to read chunk data")?;
      
      match chunk_hdr.ty {
        JSON_CHUNK => {
          if json_text.is_some() {
            return Err("multiple json chunks in glb");
          }

          json_text = Some(String::from_utf8(bytes).map_err(|_|"json chunk failed to stringify")?);
        }
        BIN_CHUNK => {
          buffers.push(bytes);
        }
        _ => {
          return Err("unrecognized glb chunk type");
        }
      }
    }

    if json_text.is_none() {
      return Err("no json chunk");
    }

    (json_text.unwrap(), Some(buffers))
  };

  let root = match json::parse(&json_text) {
    Ok(root) => root,
    Err(_msg) => {
      return Err("failed to parse gltf json:");
    }
  };

  return load_from_root(
    root.as_obj().ok_or("root not object")?,
    &buffers,
    dir
  );
}

pub
struct Contents {
  pub mesh_names: Vec<String>,
  pub meshes: Vec<Vec<renderer::CPUStaticMesh>>,
  pub nodes: Vec<SceneNode>,
  pub scenes: Vec<Vec<usize>>,
  pub root_scene: usize
}

pub fn load_from_root(root: &HashMap<String, json::Node>, preloaded_buffers: &Option<Vec<Vec<u8>>>, dir: &str) -> Result<Contents, &'static str> {
  // Verify the version

  let asset = root.get("asset").ok_or("no asset")?.as_obj().ok_or("asset not object")?;
  let version = asset.get("version").ok_or("no version")?.as_string().ok_or("version not string")?;

  if version != "2.0" {
    return Err("only gltf 2.0 assets supported");
  }

  // Load all buffers

  let mut external_buffers = Vec::<Vec<u8>>::new();

  let buffers = if let Some(buffers) = preloaded_buffers {
    buffers
  }
  else {
    for b in root.get("buffers").ok_or("no buffers")?.as_array().ok_or("buffers not array")? {
      let buf = b.as_obj().ok_or("buffer not object")?;

      let _length = buf.get("byteLength").ok_or("no buffer byteLength")?.as_integer().ok_or("buffer byteLength not integer")?;
      
      let uri = buf.get("uri").ok_or("no buffer uri")?.as_string().ok_or("buffer uri not string")?;

      let base64_prefix = "data:application/octet-stream;base64,";

      let data = if uri.starts_with(base64_prefix) {
        let chars: Vec<char> = uri.chars().collect();
        base64::decode(&chars[base64_prefix.len()..]).ok_or("failed to decode base64 buffer")?
      }
      else {
        let buf_path = format!("{}/{}", dir, uri);
        std::fs::read(buf_path).map_err(|_|"failed to load buffer")?
      };

      external_buffers.push(data);
    }

    &external_buffers
  };
  
  // Gather buffer views

  let mut buffer_views = Vec::<BufferView>::new();

  for bv in root.get("bufferViews").ok_or("no buffer views")?.as_array().ok_or("buffer views not array")? {
    let buf_view = bv.as_obj().ok_or("buffer view not object")?;

    let buf = buf_view.get("buffer").ok_or("no buffer")?.as_integer().ok_or("buffer view buffer not integer")?;
    let length = buf_view.get("byteLength").ok_or("no buffer view length")?.as_integer().ok_or("buffer view length not integer")?;

    let offset = if let Some(off) = buf_view.get("byteOffset") {
      off.as_integer().ok_or("buffer view offset not integer")?
    }
    else {
      0
    };

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

    if acc.contains_key("sparse") {
      return Err("accessor contains sparse attribute which is not yet handled");
    }

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

  // Gather scenes

  let mut scenes = Vec::<Scene>::new();

  for s in root.get("scenes").ok_or("no scenes")?.as_array().ok_or("scenes not array")? {
    let scene = s.as_obj().ok_or("scene not object")?;

    let name = if let Some(n) = scene.get("name") {
      n.as_string().ok_or("scene name not string")?.as_str()
    }
    else {
      "unnamed"
    };

    let mut nds = Vec::<usize>::new();

    for n in scene.get("nodes").ok_or("no scene nodes")?.as_array().ok_or("scene nodes not array")? {
      nds.push(n.as_integer().ok_or("scene node not integer")? as usize);
    }

    scenes.push(Scene {
      name,
      nodes: nds
    });
  }

  let mut mesh_names = Vec::<String>::new();
  let mut static_meshes = Vec::<Vec<renderer::CPUStaticMesh>>::new();

  // Process all meshes into static meshes
  for m in meshes {
    let mut prims = Vec::new();

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

      prims.push(renderer::CPUStaticMesh{
        vertex_data,
        index_data
      });

      mesh_names.push(m.name.clone());
    }

    static_meshes.push(prims);
  }

  let mut nodes = Vec::<SceneNode>::new();

  for n in root.get("nodes").ok_or("no nodes")?.as_array().ok_or("nodes not array")? {
    let node = n.as_obj().ok_or("node not object")?;

    let name = if let Some(n) = node.get("name") {
      n.as_string().ok_or("node name not string")?.clone()
    }
    else {
      String::from("unnamed")
    };

    let children: Vec<usize> = if node.contains_key("children") {
      let mut list = Vec::new();

      for c in node["children"].as_array().ok_or("node children not array")? {
        list.push(c.as_integer().ok_or("node child not integer")? as usize);
      }

      list
    }
    else {
      Vec::new()
    };

    let mesh = if node.contains_key("mesh") {
      Some(node["mesh"].as_integer().ok_or("node mesh not integer")? as usize)
    }
    else {
      None
    };

    let local_transform = if node.contains_key("matrix") {
      let mut data = [0.0 as f32;16];

      for (i , x) in node["matrix"].as_array().ok_or("matrix not array")?.iter().enumerate() {
        let val = x.as_number().ok_or("matrix entry not number")? as f32;
        data[i] = val;
      }

      Matrix4::<f32>::from_row_slice(&data).transpose()
    }
    else {

      let scale = if node.contains_key("scale") {
        as_vec::<3>(&node["scale"])?
      }
      else {
        Vector3::new(1.0, 1.0, 1.0)
      };

      let translation = if node.contains_key("translation") {
        as_vec::<3>(&node["translation"])?
      }
      else {
        Vector3::zeros()
      };

      let rotation = if node.contains_key("rotation") {
        let vec = as_vec::<4>(&node["rotation"])?;
        let quat = geometry::Quaternion::from_vector(vec);
        geometry::UnitQuaternion::from_quaternion(quat)
      }
      else {
        geometry::UnitQuaternion::identity()
      };

      let scaling_m: Matrix4<f32> = Matrix4::new_nonuniform_scaling(&scale);
      let rotation_m: Matrix4<f32> = rotation.to_rotation_matrix().to_homogeneous();
      let translation_m: Matrix4<f32> = Matrix4::new_translation(&translation);

      translation_m * rotation_m * scaling_m
    };

    nodes.push(SceneNode{
      name,
      local_transform,
      children,
      mesh
    });
  }

  std::assert!(nodes.len() == root.get("nodes").unwrap().as_array().unwrap().len());

  let mut scenes = Vec::new();

  for s in root.get("scenes").ok_or("no scenes")?.as_array().ok_or("scenes not array")? {
    let scn = s.as_obj().ok_or("scene not object")?;

    let mut scene = Vec::<usize>::new();

    for n in scn.get("nodes").ok_or("no scene nodes")?.as_array().ok_or("scene nodes not array")? {
      scene.push(n.as_integer().ok_or("scene node not integer")? as usize);
    }

    scenes.push(scene);
  }

  let root_scene = root.get("scene").ok_or("no root scene")?.as_integer().ok_or("root scene not integer")? as usize;

  return Ok(Contents{
    mesh_names,
    meshes: static_meshes,
    nodes,
    scenes,
    root_scene
  });
}

fn as_vec<const N: usize>(node: &json::Node) -> Result<Vector<f32, Const<N>, ArrayStorage<f32, N, 1>>, &'static str> {
  let s = node.as_array().ok_or("supposed vector not array")?;

  if (s.len() != N) {
    return Err("vector not properly sized");
  }

  let mut vals = Vector::<f32, Const<N>, ArrayStorage<f32, N, 1>>::zeros(); 

  for (i, x) in s.iter().enumerate() {
    vals[i] = x.as_number().ok_or("supposed vector contains non-number")? as _;
  }

  return Ok(vals);
}