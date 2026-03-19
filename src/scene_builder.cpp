#include "scene_builder.hpp"

#include <sstream>
#include <stdexcept>

namespace {
CameraData makeCamera(simd_float3 origin, simd_float3 target, float focal_length, float viewport_scale) {
    CameraData camera {};
    const simd_float3 world_up = simd_make_float3(0.0f, 1.0f, 0.0f);
    camera.origin = origin;
    camera.forward = simd_normalize(target - origin);
    camera.right = simd_normalize(simd_cross(camera.forward, world_up));
    camera.up = simd_normalize(simd_cross(camera.right, camera.forward));
    camera.focalLength = focal_length;
    camera.viewportScale = viewport_scale;
    return camera;
}

PlaneData makeGroundPlane(float y, simd_float3 a, simd_float3 b, float checker_scale, float roughness = 0.65f) {
    PlaneData plane {};
    plane.normal = simd_make_float3(0.0f, 1.0f, 0.0f);
    plane.offset = -y;
    plane.albedoA = a;
    plane.albedoB = b;
    plane.checkerScale = checker_scale;
    plane.roughness = roughness;
    plane.materialType = MATERIAL_LAMBERTIAN;
    plane.emissionStrength = 0.0f;
    return plane;
}

SphereData makeSphere(simd_float3 center, float radius, simd_float3 albedo, float roughness, MaterialType material, float emission = 0.0f) {
    SphereData sphere {};
    sphere.center = center;
    sphere.radius = radius;
    sphere.albedo = albedo;
    sphere.roughness = roughness;
    sphere.materialType = static_cast<uint32_t>(material);
    sphere.emissionStrength = emission;
    return sphere;
}

TriangleData makeTriangle(simd_float3 v0, simd_float3 v1, simd_float3 v2, simd_float3 albedo, float roughness = 0.2f, MaterialType material = MATERIAL_LAMBERTIAN, float emission = 0.0f) {
    TriangleData tri {};
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.albedo = albedo;
    tri.roughness = roughness;
    tri.materialType = static_cast<uint32_t>(material);
    tri.emissionStrength = emission;
    return tri;
}

SceneDescription makeStarterScene() {
    SceneDescription scene;
    scene.name = "starter";
    scene.label = "Starter";
    scene.description = "Starter composition with Lambertian, metal, coated, dielectric, and emissive materials, with the emissive triangle positioned to cast visible direct light.";
    scene.camera = makeCamera(simd_make_float3(0.0f, 0.9f, 2.2f), simd_make_float3(0.0f, -0.15f, -3.8f), 1.5f, 1.0f);
    scene.planes.push_back(makeGroundPlane(-1.0f, simd_make_float3(0.85f, 0.86f, 0.88f), simd_make_float3(0.28f, 0.30f, 0.34f), 1.2f));
    scene.spheres = {
        makeSphere(simd_make_float3(-1.2f, -0.15f, -3.6f), 0.85f, simd_make_float3(0.82f, 0.33f, 0.20f), 0.18f, MATERIAL_LAMBERTIAN),
        makeSphere(simd_make_float3(0.15f, -0.30f, -2.75f), 0.70f, simd_make_float3(0.92f, 0.96f, 1.00f), 0.04f, MATERIAL_DIELECTRIC),
        makeSphere(simd_make_float3(1.65f, -0.45f, -4.35f), 0.55f, simd_make_float3(0.22f, 0.52f, 0.88f), 0.22f, MATERIAL_COATED),
        makeSphere(simd_make_float3(2.45f, -0.58f, -5.10f), 0.38f, simd_make_float3(0.78f, 0.82f, 0.88f), 0.10f, MATERIAL_METAL),
    };
    scene.triangles = {
        makeTriangle(simd_make_float3(-2.4f, -0.15f, -5.8f), simd_make_float3(-1.7f, 1.0f, -5.4f), simd_make_float3(-0.9f, -0.2f, -5.6f), simd_make_float3(0.78f, 0.26f, 0.22f), 0.25f, MATERIAL_LAMBERTIAN),
        makeTriangle(simd_make_float3(0.45f, 0.15f, -3.45f), simd_make_float3(1.65f, 1.10f, -3.85f), simd_make_float3(2.05f, -0.05f, -3.10f), simd_make_float3(1.00f, 0.86f, 0.55f), 0.0f, MATERIAL_EMISSIVE, 4.25f),
    };
    return scene;
}

SceneDescription makeWideScene() {
    SceneDescription scene;
    scene.name = "wide";
    scene.label = "Wide";
    scene.description = "Wider framing with mixed Lambertian, metal, coated, dielectric, and emissive surfaces, stronger emissive lighting, hard shadows, and multisampled direct lighting.";
    scene.camera = makeCamera(simd_make_float3(0.0f, 1.35f, 3.8f), simd_make_float3(0.0f, -0.35f, -4.4f), 1.7f, 1.0f);
    scene.planes.push_back(makeGroundPlane(-1.0f, simd_make_float3(0.90f, 0.90f, 0.92f), simd_make_float3(0.22f, 0.24f, 0.28f), 1.4f));
    scene.spheres = {
        makeSphere(simd_make_float3(-2.3f, -0.35f, -5.1f), 0.65f, simd_make_float3(0.85f, 0.42f, 0.22f), 0.22f, MATERIAL_LAMBERTIAN),
        makeSphere(simd_make_float3(-0.7f, -0.10f, -4.0f), 0.90f, simd_make_float3(0.82f, 0.86f, 0.92f), 0.08f, MATERIAL_METAL),
        makeSphere(simd_make_float3(0.95f, -0.42f, -3.25f), 0.58f, simd_make_float3(0.94f, 0.98f, 1.00f), 0.03f, MATERIAL_DIELECTRIC),
        makeSphere(simd_make_float3(2.35f, -0.55f, -5.6f), 0.45f, simd_make_float3(0.28f, 0.76f, 0.55f), 0.24f, MATERIAL_COATED),
    };
    scene.triangles = {
        makeTriangle(simd_make_float3(-3.4f, -0.1f, -7.0f), simd_make_float3(-2.2f, 1.9f, -7.4f), simd_make_float3(-1.0f, -0.15f, -6.6f), simd_make_float3(0.92f, 0.32f, 0.24f), 0.22f, MATERIAL_LAMBERTIAN),
        makeTriangle(simd_make_float3(-0.2f, -0.25f, -7.2f), simd_make_float3(1.2f, 1.55f, -7.6f), simd_make_float3(2.0f, -0.2f, -6.9f), simd_make_float3(0.20f, 0.56f, 0.88f), 0.10f, MATERIAL_METAL),
        makeTriangle(simd_make_float3(1.15f, 0.10f, -4.65f), simd_make_float3(2.45f, 1.25f, -5.05f), simd_make_float3(3.05f, -0.05f, -4.30f), simd_make_float3(1.00f, 0.90f, 0.62f), 0.0f, MATERIAL_EMISSIVE, 4.85f),
    };
    return scene;
}

SceneDescription makeMaterialsScene() {
    SceneDescription scene;
    scene.name = "materials";
    scene.label = "Materials";
    scene.description = "Sphere-only material showcase with Lambertian, metal, coated, dielectric, and emissive spheres on a large ground sphere.";
    scene.camera = makeCamera(simd_make_float3(0.0f, 1.0f, 4.8f), simd_make_float3(0.0f, -0.25f, -4.6f), 1.65f, 1.0f);
    scene.spheres = {
        makeSphere(simd_make_float3(0.0f, -1001.2f, -5.5f), 1000.0f, simd_make_float3(0.78f, 0.80f, 0.84f), 0.55f, MATERIAL_LAMBERTIAN),
        makeSphere(simd_make_float3(-3.6f, -0.15f, -5.8f), 0.95f, simd_make_float3(0.82f, 0.32f, 0.20f), 0.20f, MATERIAL_LAMBERTIAN),
        makeSphere(simd_make_float3(-1.7f, -0.18f, -5.2f), 0.92f, simd_make_float3(0.78f, 0.82f, 0.88f), 0.06f, MATERIAL_METAL),
        makeSphere(simd_make_float3(0.1f, -0.20f, -4.9f), 0.90f, simd_make_float3(0.20f, 0.54f, 0.90f), 0.18f, MATERIAL_COATED),
        makeSphere(simd_make_float3(1.9f, -0.18f, -5.1f), 0.92f, simd_make_float3(0.94f, 0.98f, 1.00f), 0.03f, MATERIAL_DIELECTRIC),
        makeSphere(simd_make_float3(3.7f, -0.12f, -5.7f), 0.98f, simd_make_float3(1.00f, 0.86f, 0.56f), 0.0f, MATERIAL_EMISSIVE, 3.8f),
        makeSphere(simd_make_float3(-0.85f, 1.85f, -6.8f), 0.65f, simd_make_float3(1.00f, 0.95f, 0.82f), 0.0f, MATERIAL_EMISSIVE, 4.8f),
        makeSphere(simd_make_float3(2.2f, 1.45f, -6.4f), 0.50f, simd_make_float3(0.72f, 0.84f, 1.00f), 0.0f, MATERIAL_EMISSIVE, 3.4f),
    };
    return scene;
}

SceneDescription makeRoughnessScene() {
    SceneDescription scene;
    scene.name = "roughness";
    scene.label = "Roughness";
    scene.description = "Roughness comparison scene with checker ground and a dark backdrop, showing metal and dielectric spheres from smooth to rough under emissive sphere lighting.";
    scene.camera = makeCamera(simd_make_float3(0.0f, 1.0f, 4.6f), simd_make_float3(0.0f, 0.15f, -4.9f), 1.95f, 0.95f);
    scene.planes = {
        makeGroundPlane(-1.70f, simd_make_float3(0.90f, 0.90f, 0.92f), simd_make_float3(0.30f, 0.32f, 0.36f), 1.7f, 0.35f),
    };
    scene.spheres = {
        makeSphere(simd_make_float3(-3.4f, 1.30f, -5.9f), 0.95f, simd_make_float3(0.82f, 0.84f, 0.88f), 0.02f, MATERIAL_METAL),
        makeSphere(simd_make_float3(-1.15f, 1.30f, -5.3f), 0.95f, simd_make_float3(0.82f, 0.84f, 0.88f), 0.18f, MATERIAL_METAL),
        makeSphere(simd_make_float3(1.15f, 1.30f, -5.3f), 0.95f, simd_make_float3(0.82f, 0.84f, 0.88f), 0.38f, MATERIAL_METAL),
        makeSphere(simd_make_float3(3.4f, 1.30f, -5.9f), 0.95f, simd_make_float3(0.82f, 0.84f, 0.88f), 0.62f, MATERIAL_METAL),

        makeSphere(simd_make_float3(-3.4f, -0.75f, -5.9f), 0.95f, simd_make_float3(0.88f, 0.95f, 1.00f), 0.01f, MATERIAL_DIELECTRIC),
        makeSphere(simd_make_float3(-1.15f, -0.75f, -5.3f), 0.95f, simd_make_float3(0.88f, 0.95f, 1.00f), 0.12f, MATERIAL_DIELECTRIC),
        makeSphere(simd_make_float3(1.15f, -0.75f, -5.3f), 0.95f, simd_make_float3(0.88f, 0.95f, 1.00f), 0.26f, MATERIAL_DIELECTRIC),
        makeSphere(simd_make_float3(3.4f, -0.75f, -5.9f), 0.95f, simd_make_float3(0.88f, 0.95f, 1.00f), 0.44f, MATERIAL_DIELECTRIC),

        makeSphere(simd_make_float3(-2.15f, 3.15f, -6.4f), 0.78f, simd_make_float3(1.00f, 0.94f, 0.80f), 0.0f, MATERIAL_EMISSIVE, 4.8f),
        makeSphere(simd_make_float3(2.35f, 2.95f, -6.0f), 0.64f, simd_make_float3(0.76f, 0.86f, 1.00f), 0.0f, MATERIAL_EMISSIVE, 3.6f),
        makeSphere(simd_make_float3(-2.25f, -0.10f, -9.4f), 0.55f, simd_make_float3(0.86f, 0.22f, 0.18f), 0.18f, MATERIAL_LAMBERTIAN),
        makeSphere(simd_make_float3(2.15f, -0.05f, -9.0f), 0.60f, simd_make_float3(0.18f, 0.46f, 0.84f), 0.22f, MATERIAL_COATED),
    };
    scene.triangles = {
        makeTriangle(simd_make_float3(-6.4f, -1.75f, -11.0f), simd_make_float3(-6.4f, 5.0f, -11.0f), simd_make_float3(6.4f, 5.0f, -11.0f), simd_make_float3(0.17f, 0.18f, 0.22f), 0.75f, MATERIAL_LAMBERTIAN),
        makeTriangle(simd_make_float3(-6.4f, -1.75f, -11.0f), simd_make_float3(6.4f, 5.0f, -11.0f), simd_make_float3(6.4f, -1.75f, -11.0f), simd_make_float3(0.17f, 0.18f, 0.22f), 0.75f, MATERIAL_LAMBERTIAN),
    };
    return scene;
}

std::string escapeJson(const std::string& text) {
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            default: out << c; break;
        }
    }
    return out.str();
}
}  // namespace

SceneDescription buildScene(const std::string& scene_name) {
    if (scene_name == "starter") return makeStarterScene();
    if (scene_name == "wide") return makeWideScene();
    if (scene_name == "materials") return makeMaterialsScene();
    if (scene_name == "roughness") return makeRoughnessScene();
    throw std::runtime_error("Unknown scene: " + scene_name);
}

std::vector<std::string> availableSceneNames() {
    return {"starter", "wide", "materials", "roughness"};
}

std::string buildOptionsSchemaJson() {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema_version\": 3,\n";
    out << "  \"render\": {\n";
    out << "    \"width\": 960,\n";
    out << "    \"height\": 540,\n";
    out << "    \"spp\": 16,\n";
    out << "    \"adaptive_min_spp\": 8,\n";
    out << "    \"firefly_clamp\": 6.0,\n";
    out << "    \"denoise_strength\": 1.0,\n";
    out << "    \"adaptive_threshold\": 0.02\n";
    out << "  },\n";
    out << "  \"choices\": {\n";
    out << "    \"scene\": [\"starter\", \"wide\", \"materials\", \"roughness\"]\n";
    out << "  },\n";
    out << "  \"defaults\": {\n";
    out << "    \"scene\": \"starter\",\n";
    out << "    \"output\": \"outputs/metal_starter.ppm\",\n";
    out << "    \"denoise\": true,\n";
    out << "    \"adaptive_sampling\": false,\n";
    out << "    \"save_guide_buffers\": false\n";
    out << "  },\n";
    out << "  \"capabilities\": {\n";
    out << "    \"scene_registry\": true,\n";
    out << "    \"save_png\": false,\n";
    out << "    \"save_guide_buffers\": true,\n";
    out << "    \"adaptive_sampling\": true,\n";
    out << "    \"material_types\": [\"lambertian\", \"metal\", \"emissive\", \"dielectric\", \"coated\"],\n";
    out << "    \"hard_shadows\": true,\n";
    out << "    \"multisampling\": true,\n";
    out << "    \"softer_directional_light\": true,\n";
    out << "    \"denoiser\": \"gpu_atrous_lobe_aware\",\n";
    out << "    \"firefly_clamping\": true\n";
    out << "  }\n";
    out << "}\n";
    return out.str();
}

std::string buildSceneRegistryJson() {
    const auto scenes = std::vector<SceneDescription>{makeStarterScene(), makeWideScene(), makeMaterialsScene(), makeRoughnessScene()};
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"default_scene\": \"starter\",\n";
    out << "  \"scenes\": [\n";
    for (size_t i = 0; i < scenes.size(); ++i) {
        const auto& scene = scenes[i];
        out << "    {\n";
        out << "      \"name\": \"" << escapeJson(scene.name) << "\",\n";
        out << "      \"label\": \"" << escapeJson(scene.label) << "\",\n";
        out << "      \"description\": \"" << escapeJson(scene.description) << "\",\n";
        out << "      \"primitive_counts\": {\n";
        out << "        \"spheres\": " << scene.spheres.size() << ",\n";
        out << "        \"planes\": " << scene.planes.size() << ",\n";
        out << "        \"triangles\": " << scene.triangles.size() << "\n";
        out << "      }\n";
        out << "    }" << (i + 1 < scenes.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}
