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
}

StructuredBuffer<Vertex> vbuffers[] : register(t0, space0); 
StructuredBuffer<uint> ibuffers[] : register(t0, space1); 

VSOut vs_main(uint vid : SV_VertexID) {
  uint index = ibuffers[ibuffer_index][vid];
  Vertex vertex = vbuffers[vbuffer_index][index];

  VSOut vso;
  vso.sv_pos = mul(float4(vertex.pos, 1.0f), view_proj);
  vso.norm = normalize(vertex.norm);

  return vso;
}

float4 ps_main(VSOut vso) : SV_Target {
  float3 norm = normalize(vso.norm);
  return float4(sqrt(norm * 0.5f + 0.5f), 1.0f);
}
