struct Vertex {
  float3 pos;
  float3 norm;
};

static Vertex vbuffer[] = {
  {{ 0.0f,  0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
  {{-0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
  {{ 0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}},
};

struct VSOut {
  float4 sv_pos : SV_Position;
  float3 color : Color;
};

cbuffer Camera : register(b0, space0) {
  float4x4 view_proj;
};

VSOut vs_main(uint vid : SV_VertexID) {
  Vertex vertex = vbuffer[vid];

  VSOut vso;
  vso.sv_pos = mul(float4(vertex.pos, 1.0f), view_proj);
  vso.color = vertex.norm;

  return vso;
}

float4 ps_main(VSOut vso) : SV_Target {
  return float4(sqrt(vso.color), 1.0f);
}
