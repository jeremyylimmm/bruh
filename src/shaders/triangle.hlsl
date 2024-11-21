struct Vertex {
  float3 pos;
  float3 norm;
  float2 tex_coord;
};

struct VSOut {
  float4 sv_pos : SV_Position;
  float3 norm : Normal;
};

cbuffer Camera : register(b0, space0) {
  float4x4 view_proj;
};

cbuffer InstanceData : register(b1, space0) {
  uint vbuffer_index;
  uint ibuffer_index;
  uint transform_index;
}

struct Transform {
  float4x4 m;
};

StructuredBuffer<Vertex> vbuffers[] : register(t0, space0); 
StructuredBuffer<uint> ibuffers[] : register(t0, space1); 
ConstantBuffer<Transform> transforms[] : register(b0, space2);

VSOut vs_main(uint vid : SV_VertexID) {
  uint index = ibuffers[ibuffer_index][vid];
  Vertex vertex = vbuffers[vbuffer_index][index];

  float4x4 transform = transforms[transform_index].m;

  VSOut vso;
  vso.sv_pos = mul(view_proj, mul(transform, float4(vertex.pos, 1.0f) ));
  vso.norm = normalize(mul(transform, float4(vertex.norm, 0.0f)));

  return vso;
}

float4 ps_main(VSOut vso) : SV_Target {
  float3 norm = normalize(vso.norm);

  float3 light_dir = normalize(float3(-1.0, 1.0, 1.0));
  float3 diffuse = max(dot(norm, light_dir), 0.0).xxx;

  return float4(sqrt(diffuse), 1.0f);
}
