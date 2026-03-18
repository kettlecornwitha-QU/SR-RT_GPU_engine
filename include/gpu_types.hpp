#pragma once

#include <simd/simd.h>

#include <cstdint>
#include <string>
#include <vector>

enum MaterialType : uint32_t {
    MATERIAL_LAMBERTIAN = 0,
    MATERIAL_METAL = 1,
    MATERIAL_EMISSIVE = 2,
    MATERIAL_DIELECTRIC = 3,
    MATERIAL_COATED = 4,
};

struct RenderOptions {
    uint32_t width = 960;
    uint32_t height = 540;
    uint32_t spp = 16;
    std::string output = "outputs/metal_starter.ppm";
    std::string scene = "starter";
    bool print_options_schema = false;
    bool print_scene_registry = false;
};

struct RenderUniforms {
    uint32_t width;
    uint32_t height;
    uint32_t sphereCount;
    uint32_t planeCount;
    uint32_t triangleCount;
    uint32_t samplesPerPixel;
    uint32_t sampleIndex;
    float time;
    float lightAngularRadius;
    simd_float3 pad0;
};

struct CameraData {
    simd_float3 origin;
    float pad0;
    simd_float3 forward;
    float pad1;
    simd_float3 right;
    float pad2;
    simd_float3 up;
    float focalLength;
    float viewportScale;
    simd_float3 pad3;
};

struct SphereData {
    simd_float3 center;
    float radius;
    simd_float3 albedo;
    float roughness;
    uint32_t materialType;
    float emissionStrength;
    simd_float2 pad0;
};

struct PlaneData {
    simd_float3 normal;
    float offset;
    simd_float3 albedoA;
    float checkerScale;
    simd_float3 albedoB;
    float roughness;
    uint32_t materialType;
    float emissionStrength;
    simd_float2 pad0;
};

struct TriangleData {
    simd_float3 v0;
    float pad0;
    simd_float3 v1;
    float pad1;
    simd_float3 v2;
    float pad2;
    simd_float3 albedo;
    float roughness;
    uint32_t materialType;
    float emissionStrength;
    simd_float2 pad3;
};

struct SceneDescription {
    std::string name;
    std::string label;
    std::string description;
    CameraData camera {};
    std::vector<SphereData> spheres;
    std::vector<PlaneData> planes;
    std::vector<TriangleData> triangles;
};
