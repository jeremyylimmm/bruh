 struct Vertex {
  float3 pos;
  float pad;
};

cbuffer Camera : register(b0, space0) {
  float4x4 view_proj;
};

StructuredBuffer<Vertex> vbuffers : register(t0, space0); 

float4 vs_main(uint vid : SV_VertexID) : SV_Position {
  Vertex vertex = vbuffers[vid];

  return mul(view_proj, float4(vertex.pos, 1.0f));
}

float4 ps_main() : SV_Target {
  return float4(1.0f, 0.0f, 0.0f, 1.0f);
}