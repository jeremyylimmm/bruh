#[derive(Copy, Clone, Debug)]
pub struct FloatNxN<const R: usize, const C: usize> {
  m: [[f32;C];R]
}

#[allow(dead_code)]
pub type Float4x4 = FloatNxN<4, 4>;

#[allow(dead_code)]
impl<const R: usize, const C: usize> FloatNxN<R, C> {
  pub fn from(data: [[f32;C];R]) -> Self {
    return Self {
      m: data
    };
  }

  pub fn identity() -> Self {
    return Self {
      m: std::array::from_fn(|i|std::array::from_fn(|j|if i == j {1.0} else {0.0}))
    }
  }

  fn column(&self, index: usize) -> [f32;R] {
    return std::array::from_fn(|i|self.m[i][index]);
  }
}

impl<const N: usize> FloatNxN<N, N> {
  pub fn inverse(&self) -> Option<Self> {
    let mut this = Vec::<f32>::new();

    for i in 0..N {
      for j in 0..N {
        this.push(self.m[i][j]);
      }
    }

    let mut masked = Vec::<f32>::new();

    let dets = Self {
      m: std::array::from_fn(|i|std::array::from_fn(|j|
        Self::submatrix_det(&this, &mut masked, N, j, i)
      ))
    };

    let row: [f32;N] = std::array::from_fn(|i|dets.m[i][0] * self.m[0][i]);
    let det: f32 = row.iter().sum();

    if det == 0.0 {
      return None;
    } else {
      return Some(dets * (1.0 / det));
    }
  }

  pub fn submatrix_det(this: &Vec<f32>, masked: &mut Vec<f32>, n: usize, i: usize, j: usize) -> f32 {
    let sign = if (i+j) % 2 == 0 { 1.0 } else { -1.0 };
    mask(this, masked, n, i, j);
    return sign * det_raw(masked, n-1);
  }
}

fn mask(cur: &Vec<f32>, result: &mut Vec<f32>, n: usize, mask_i: usize, mask_j: usize) {
  result.clear();

  for i in 0..n {
    for j in 0..n {
      if i != mask_i && j != mask_j {
        result.push(cur[i*n+j]);
      }
    }
  }
}

fn det_raw(d: &Vec<f32>, n: usize) -> f32 {
  if n == 2 {
    return d[0]*d[3] - d[2]*d[1]
  }
  else {
    let mut det = 0.0;

    let signs = [1.0, -1.0];
    let mut sign_iter = signs.iter().cycle();

    let mut masked = Vec::<f32>::new();

    for i in 0..n {
      masked.clear();
      mask(d, &mut masked, n, 0, i);

      let sign = sign_iter.next().unwrap();
      det += sign * d[i] * det_raw(&masked, n-1);
    }

    return det;
  }
}

impl<const R1: usize, const C1: usize, const C2: usize> std::ops::Mul<FloatNxN<C1, C2>> for FloatNxN<R1, C1> {
  type Output = FloatNxN<R1, C2>;

  fn mul(self, rhs: FloatNxN<C1, C2>) -> Self::Output {
    return Self::Output {
      m: std::array::from_fn(|i|std::array::from_fn(|j| dot_product(self.m[i], rhs.column(j))))
    };
  }
}

impl<const R: usize, const C: usize> std::ops::Mul<f32> for FloatNxN<R, C> {
  type Output = Self;

  fn mul(self, rhs: f32) -> Self::Output {
    return Self::Output {
      m: std::array::from_fn(|i|std::array::from_fn(|j|
        self.m[i][j]*rhs
      ))
    };
  }
} 

fn dot_product<const N: usize>(a: [f32;N], b: [f32;N]) -> f32 {
  let mut result = 0.0;

  for i in 0..N {
    result += a[i] * b[i];
  }

  return result;
}

pub fn translation(v: &[f32;3]) -> Float4x4 {
  let mut result = Float4x4::identity();

  result.m[0][3] = v[0];
  result.m[1][3] = v[1];
  result.m[2][3] = v[2];

  return result;
}

pub fn scaling(v: &[f32;3]) -> Float4x4 {
  let mut result = Float4x4::identity();

  result.m[0][0] = v[0];
  result.m[1][1] = v[1];
  result.m[2][2] = v[2];

  return result;
}

pub fn quaternion_roll_pitch_yaw(roll: f32, pitch: f32, yaw: f32) -> [f32;4] {
  let halfpitch = pitch * 0.5;
  let cp = halfpitch.cos();
  let sp = halfpitch.sin();

  let halfyaw = yaw * 0.5;
  let cy = halfyaw.cos();
  let sy = halfyaw.sin();

  let halfroll = roll * 0.5;
  let cr = halfroll.cos();
  let sr = halfroll.sin();

  return [
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    sr * cp * cy - cr * sp * sy,
    cr * cp * cy + sr * sp * sy
  ];
}

pub fn rotation(v: &[f32;4]) -> Float4x4 {
  let qx = v[0];
  let qxx = qx * qx;

  let qy = v[1];
  let qyy = qy * qy;

  let qz = v[2];
  let qzz = qz * qz;

  let qw = v[3];

  let mut result = Float4x4::identity();

  result.m[0][0] = 1.0 - 2.0 * qyy - 2.0 * qzz;
  result.m[1][0] = 2.0 * qx * qy + 2.0 * qz * qw;
  result.m[2][0] = 2.0 * qx * qz - 2.0 * qy * qw;
  result.m[3][0] = 0.0;
             
  result.m[0][1] = 2.0 * qx * qy - 2.0 * qz * qw;
  result.m[1][1] = 1.0 - 2.0 * qxx - 2.0 * qzz;
  result.m[2][1] = 2.0 * qy * qz + 2.0 * qx * qw;
  result.m[3][1] = 0.0;
             
  result.m[0][2] = 2.0 * qx * qz + 2.0 * qy * qw;
  result.m[1][2] = 2.0 * qy * qz - 2.0 * qx * qw;
  result.m[2][2] = 1.0 - 2.0 * qxx - 2.0 * qyy;
  result.m[3][2] = 0.0;
             
  result.m[0][3] = 0.0;
  result.m[1][3] = 0.0;
  result.m[2][3] = 0.0;
  result.m[3][3] = 1.0;

  return result;
}

pub fn perspective_rh(fov_y: f32, aspect: f32, near_z: f32, far_z: f32) -> Float4x4 {
  let sin_fov = (fov_y * 0.5).sin();
  let cos_fov = (fov_y * 0.5).cos();

  let height = cos_fov / sin_fov;
  let width = height / aspect;
  let f_range = far_z / (near_z - far_z);

  let mut result = Float4x4::identity();

  result.m[0][0] = width;
  result.m[1][0] = 0.0;
  result.m[2][0] = 0.0;
  result.m[3][0] = 0.0;
             
  result.m[0][1] = 0.0;
  result.m[1][1] = height;
  result.m[2][1] = 0.0;
  result.m[3][1] = 0.0;
             
  result.m[0][2] = 0.0;
  result.m[1][2] = 0.0;
  result.m[2][2] = f_range;
  result.m[3][2] = -1.0;
             
  result.m[0][3] = 0.0;
  result.m[1][3] = 0.0;
  result.m[2][3] = f_range * near_z;
  result.m[3][3] = 0.0;

  return result;
}