fn main() -> std::result::Result<(), String> {
  let out_dir = "shaders";

  let shaders = std::fs::read_dir("src/shaders").unwrap();

  for path_res in shaders {
    let path = path_res.unwrap();

    if !path.file_type().unwrap().is_file() {
      continue;
    }

    compile_shader(out_dir, &path.path(), &ShaderKind::Vertex)?;
    compile_shader(out_dir, &path.path(), &ShaderKind::Pixel)?;
  }

  return Ok(());
}

enum ShaderKind {
  Vertex,
  Pixel
}

impl ShaderKind {
  fn target(&self) -> &str {
    return match self {
      Self::Vertex => { "vs_6_0" },
      Self::Pixel => { "ps_6_0" },
    };
  }

  fn entry(&self) -> &str {
    return match self {
      Self::Vertex => { "vs_main" },
      Self::Pixel => { "ps_main" },
    };
  }

  fn bin_extension(&self) -> &str {
    return match self {
      Self::Vertex => { "vso" },
      Self::Pixel => { "pso" },
    };
  }
}

fn compile_shader(out_dir: &str, path: &std::path::PathBuf, kind: &ShaderKind) -> Result<(), String> {
    let name = path.file_stem().unwrap().to_str().unwrap();

    let status = std::process::Command::new("dxc").args(&[
      path.as_os_str().to_str().unwrap(),
      "-T", kind.target(),
      "-E", kind.entry(),
      "-Fo", &format!("{}/{}.{}", out_dir, name, kind.bin_extension())
    ]).status().unwrap();

    if !status.success() {
      return Err(format!("Failed to compile {}", path.display()));
    }

    return Ok(());
}