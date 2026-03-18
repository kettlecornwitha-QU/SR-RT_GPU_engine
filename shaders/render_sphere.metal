#include <metal_stdlib>
using namespace metal;

constant uint MATERIAL_LAMBERTIAN = 0;
constant uint MATERIAL_METAL = 1;
constant uint MATERIAL_EMISSIVE = 2;
constant float PI_F = 3.14159265358979323846f;

struct RenderUniforms {
    uint width;
    uint height;
    uint sphereCount;
    uint planeCount;
    uint triangleCount;
    uint samplesPerPixel;
    uint sampleIndex;
    float time;
    float lightAngularRadius;
    float3 pad0;
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
    uint materialType;
    float emissionStrength;
    float2 pad0;
};

struct PlaneData {
    float3 normal;
    float offset;
    float3 albedoA;
    float checkerScale;
    float3 albedoB;
    float roughness;
    uint materialType;
    float emissionStrength;
    float2 pad0;
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
    uint materialType;
    float emissionStrength;
    float2 pad3;
};

struct HitInfo {
    float t;
    float3 position;
    float3 normal;
    float3 albedo;
    float roughness;
    uint materialType;
    float emissionStrength;
    bool hit;
};

static inline void init_hit(thread HitInfo& hit) {
    hit.t = 0.0f;
    hit.position = float3(0.0f);
    hit.normal = float3(0.0f);
    hit.albedo = float3(0.0f);
    hit.roughness = 0.5f;
    hit.materialType = MATERIAL_LAMBERTIAN;
    hit.emissionStrength = 0.0f;
    hit.hit = false;
}

static inline uint hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static inline float rand01(thread uint& state) {
    state = hash_u32(state + 0x9e3779b9u);
    return float(state & 0x00ffffffu) / float(0x01000000u);
}

static inline float2 rand2(thread uint& state) {
    return float2(rand01(state), rand01(state));
}

static inline void build_basis(float3 n, thread float3& t, thread float3& b) {
    float3 up = abs(n.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    t = normalize(cross(up, n));
    b = normalize(cross(n, t));
}

static inline float3 sample_directional_light(float3 baseDir, float angularRadius, thread uint& rng) {
    if (angularRadius <= 0.0f) {
        return normalize(baseDir);
    }
    float2 xi = rand2(rng);
    float r = angularRadius * sqrt(xi.x);
    float phi = 2.0f * PI_F * xi.y;
    float2 disk = r * float2(cos(phi), sin(phi));

    float3 tangent;
    float3 bitangent;
    build_basis(baseDir, tangent, bitangent);
    return normalize(baseDir + tangent * disk.x + bitangent * disk.y);
}

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
        hit.roughness = sphere.roughness;
        hit.materialType = sphere.materialType;
        hit.emissionStrength = sphere.emissionStrength;
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
        hit.roughness = plane.roughness;
        hit.materialType = plane.materialType;
        hit.emissionStrength = plane.emissionStrength;
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
        hit.roughness = tri.roughness;
        hit.materialType = tri.materialType;
        hit.emissionStrength = tri.emissionStrength;
    }
    return true;
}

static inline bool any_occluder(float3 ro, float3 rd,
                                constant SphereData* spheres, uint sphereCount,
                                constant PlaneData* planes, uint planeCount,
                                constant TriangleData* triangles, uint triangleCount,
                                float maxDistance) {
    HitInfo hit;
    init_hit(hit);
    for (uint i = 0; i < sphereCount; ++i) {
        intersect_sphere(ro, rd, spheres[i], hit);
        if (hit.hit && hit.t < maxDistance) return true;
    }
    for (uint i = 0; i < planeCount; ++i) {
        intersect_plane(ro, rd, planes[i], hit);
        if (hit.hit && hit.t < maxDistance) return true;
    }
    for (uint i = 0; i < triangleCount; ++i) {
        intersect_triangle(ro, rd, triangles[i], hit);
        if (hit.hit && hit.t < maxDistance) return true;
    }
    return false;
}

static inline float3 sky_color(float3 dir) {
    float t = clamp(0.5f * (dir.y + 1.0f), 0.0f, 1.0f);
    return mix(float3(0.96f, 0.97f, 1.0f), float3(0.46f, 0.66f, 0.94f), t);
}

static inline float3 ambient_fill(thread const HitInfo& hit) {
    float hemi = clamp(hit.normal.y * 0.5f + 0.5f, 0.0f, 1.0f);
    float3 coolSky = float3(0.30f, 0.38f, 0.48f);
    float3 warmGround = float3(0.16f, 0.14f, 0.10f);
    float3 hemiTint = mix(warmGround, coolSky, hemi);
    return hit.albedo * hemiTint * 0.28f;
}

static inline float3 shade_hit(thread const HitInfo& hit,
                               float3 rd,
                               constant SphereData* spheres, uint sphereCount,
                               constant PlaneData* planes, uint planeCount,
                               constant TriangleData* triangles, uint triangleCount,
                               constant RenderUniforms& uniforms,
                               thread uint& rng) {
    if (hit.materialType == MATERIAL_EMISSIVE) {
        return hit.albedo * hit.emissionStrength;
    }

    float3 baseLightDir = normalize(float3(-0.55f, 0.88f, -0.32f));
    float3 lightDir = sample_directional_light(baseLightDir, uniforms.lightAngularRadius, rng);
    float3 shadowOrigin = hit.position + hit.normal * 1e-3f;
    bool shadowed = any_occluder(shadowOrigin, lightDir,
                                 spheres, sphereCount,
                                 planes, planeCount,
                                 triangles, triangleCount,
                                 1e6f);

    float diffuse = max(dot(hit.normal, lightDir), 0.0f);
    float visibility = shadowed ? 0.10f : 1.0f;
    float wrapDiffuse = max((dot(hit.normal, lightDir) + 0.25f) / 1.25f, 0.0f);
    float3 directSun = float3(1.05f, 0.99f, 0.92f) * diffuse * visibility;
    float3 bouncedAmbient = ambient_fill(hit);

    if (hit.materialType == MATERIAL_LAMBERTIAN) {
        float3 base = hit.albedo * (0.70f * directSun + 0.22f * wrapDiffuse + bouncedAmbient);
        return base;
    }

    float3 viewDir = normalize(-rd);
    float3 reflected = reflect(-lightDir, hit.normal);
    float shininess = mix(90.0f, 10.0f, clamp(hit.roughness, 0.0f, 1.0f));
    float specular = pow(max(dot(viewDir, reflected), 0.0f), shininess);
    float fresnel = pow(1.0f - max(dot(viewDir, hit.normal), 0.0f), 5.0f);
    float3 reflectedSky = sky_color(reflect(rd, hit.normal));
    float3 metalBase = hit.albedo * (0.18f + 0.30f * bouncedAmbient + 0.16f * wrapDiffuse * visibility);
    float3 metalSpec = mix(hit.albedo, float3(1.0f), 0.45f) * (1.6f * specular * visibility);
    float3 environment = reflectedSky * (0.18f + 0.35f * fresnel) * (1.0f - hit.roughness * 0.55f);
    return metalBase + metalSpec + environment;
}

kernel void renderScene(device float4* outAccum [[buffer(0)]],
                        constant RenderUniforms& uniforms [[buffer(1)]],
                        constant CameraData& camera [[buffer(2)]],
                        constant SphereData* spheres [[buffer(3)]],
                        constant PlaneData* planes [[buffer(4)]],
                        constant TriangleData* triangles [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) return;

    uint rng = hash_u32(gid.x + gid.y * uniforms.width + uniforms.sampleIndex * 9781u + 0x68bc21ebu);
    float2 jitter = rand2(rng) - 0.5f;
    float2 frag = float2(gid) + 0.5f + jitter;
    float2 uv = frag / float2(uniforms.width, uniforms.height);
    float2 screen = uv * 2.0f - 1.0f;
    screen.y = -screen.y;
    screen.x *= float(uniforms.width) / float(uniforms.height);

    float3 ro = camera.origin;
    float3 rd = normalize(camera.forward * camera.focalLength + camera.right * screen.x + camera.up * (screen.y * camera.viewportScale));

    HitInfo hit;
    init_hit(hit);

    for (uint i = 0; i < uniforms.sphereCount; ++i) intersect_sphere(ro, rd, spheres[i], hit);
    for (uint i = 0; i < uniforms.planeCount; ++i) intersect_plane(ro, rd, planes[i], hit);
    for (uint i = 0; i < uniforms.triangleCount; ++i) intersect_triangle(ro, rd, triangles[i], hit);

    float3 color = hit.hit
        ? shade_hit(hit, rd, spheres, uniforms.sphereCount, planes, uniforms.planeCount, triangles, uniforms.triangleCount, uniforms, rng)
        : sky_color(rd);

    uint idx = gid.y * uniforms.width + gid.x;
    outAccum[idx] += float4(max(color, 0.0f), 1.0f);
}
