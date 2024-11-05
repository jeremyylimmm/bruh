struct Vertex {
  float3 pos;
  float3 norm;
  float2 tex_coord;
};

struct VSOut {
  float4 sv_pos : SV_Position;
  float3 color : Color;
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
  vso.color = vertex.norm;

  return vso;
}

float4 ps_main(VSOut vso) : SV_Target {
  return float4(sqrt(vso.color), 1.0f);
}
