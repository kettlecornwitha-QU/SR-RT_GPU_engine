#include <metal_stdlib>
using namespace metal;

struct RenderUniforms {
    uint width;
    uint height;
    uint sphereCount;
    uint planeCount;
    uint triangleCount;
    float time;
    float2 pad0;
};

struct CameraData {
    float3 origin;
    float pad0;
    float3 forward;
    float pad1;
    float3 right;
    float pad2;
    float3 up;
    float focalLength;
    float viewportScale;
    float3 pad3;
};

struct SphereData {
    float3 center;
    float radius;
    float3 albedo;
    float roughness;
};

struct PlaneData {
    float3 normal;
    float offset;
    float3 albedoA;
    float checkerScale;
    float3 albedoB;
    float pad0;
};

struct TriangleData {
    float3 v0;
    float pad0;
    float3 v1;
    float pad1;
    float3 v2;
    float pad2;
    float3 albedo;
    float roughness;
};

struct HitInfo {
    float t;
    float3 position;
    float3 normal;
    float3 albedo;
    bool hit;
};

static inline bool intersect_sphere(float3 ro, float3 rd, constant SphereData& sphere, thread HitInfo& hit) {
    float3 oc = ro - sphere.center;
    float a = dot(rd, rd);
    float b = 2.0f * dot(oc, rd);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant <= 0.0f) return false;
    float sqrtDisc = sqrt(discriminant);
    float t0 = (-b - sqrtDisc) / (2.0f * a);
    float t1 = (-b + sqrtDisc) / (2.0f * a);
    float t = t0 > 0.0f ? t0 : t1;
    if (t <= 0.0f) return false;
    if (!hit.hit || t < hit.t) {
        hit.hit = true;
        hit.t = t;
        hit.position = ro + rd * t;
        hit.normal = normalize(hit.position - sphere.center);
        hit.albedo = sphere.albedo;
    }
    return true;
}

static inline bool intersect_plane(float3 ro, float3 rd, constant PlaneData& plane, thread HitInfo& hit) {
    float denom = dot(plane.normal, rd);
    if (abs(denom) < 1e-5f) return false;
    float t = -(dot(plane.normal, ro) + plane.offset) / denom;
    if (t <= 0.0f) return false;
    if (!hit.hit || t < hit.t) {
        float3 p = ro + rd * t;
        float checker = floor(p.x * plane.checkerScale) + floor(p.z * plane.checkerScale);
        bool useA = fmod(checker, 2.0f) == 0.0f;
        hit.hit = true;
        hit.t = t;
        hit.position = p;
        hit.normal = plane.normal;
        hit.albedo = useA ? plane.albedoA : plane.albedoB;
    }
    return true;
}

static inline bool intersect_triangle(float3 ro, float3 rd, constant TriangleData& tri, thread HitInfo& hit) {
    const float eps = 1e-6f;
    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 p = cross(rd, e2);
    float det = dot(e1, p);
    if (abs(det) < eps) return false;
    float invDet = 1.0f / det;
    float3 tvec = ro - tri.v0;
    float u = dot(tvec, p) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    float3 qvec = cross(tvec, e1);
    float v = dot(rd, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = dot(e2, qvec) * invDet;
    if (t <= eps) return false;
    if (!hit.hit || t < hit.t) {
        hit.hit = true;
        hit.t = t;
        hit.position = ro + rd * t;
        hit.normal = normalize(cross(e1, e2));
        hit.albedo = tri.albedo;
    }
    return true;
}

kernel void renderScene(device uchar4* outPixels [[buffer(0)]],
                        constant RenderUniforms& uniforms [[buffer(1)]],
                        constant CameraData& camera [[buffer(2)]],
                        constant SphereData* spheres [[buffer(3)]],
                        constant PlaneData* planes [[buffer(4)]],
                        constant TriangleData* triangles [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) return;

    float2 frag = float2(gid) + 0.5f;
    float2 uv = frag / float2(uniforms.width, uniforms.height);
    float2 screen = uv * 2.0f - 1.0f;
    screen.y = -screen.y;
    screen.x *= float(uniforms.width) / float(uniforms.height);

    float3 ro = camera.origin;
    float3 rd = normalize(camera.forward * camera.focalLength + camera.right * screen.x + camera.up * (screen.y * camera.viewportScale));

    HitInfo hit;
    hit.t = 0.0f;
    hit.position = float3(0.0f);
    hit.normal = float3(0.0f);
    hit.albedo = float3(0.0f);
    hit.hit = false;

    for (uint i = 0; i < uniforms.sphereCount; ++i) intersect_sphere(ro, rd, spheres[i], hit);
    for (uint i = 0; i < uniforms.planeCount; ++i) intersect_plane(ro, rd, planes[i], hit);
    for (uint i = 0; i < uniforms.triangleCount; ++i) intersect_triangle(ro, rd, triangles[i], hit);

    float3 color;
    if (hit.hit) {
        float3 lightDir = normalize(float3(-0.6f, 0.9f, -0.4f));
        float diffuse = max(dot(hit.normal, lightDir), 0.0f);
        float rim = pow(1.0f - max(dot(hit.normal, -rd), 0.0f), 3.0f);
        float3 fill = float3(0.10f, 0.11f, 0.13f);
        color = hit.albedo * (fill + diffuse) + rim * float3(0.12f, 0.10f, 0.08f);
    } else {
        float t = 0.5f * (rd.y + 1.0f);
        color = mix(float3(0.95f, 0.97f, 1.0f), float3(0.40f, 0.60f, 0.90f), t);
    }

    color = clamp(color, 0.0f, 1.0f);
    uint idx = gid.y * uniforms.width + gid.x;
    uchar3 rgb = uchar3(color * 255.0f);
    outPixels[idx] = uchar4(rgb, 255);
}
