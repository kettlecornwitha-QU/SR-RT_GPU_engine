#include "gpu_types.hpp"
#include "renderer_host.hpp"
#include "scene_builder.hpp"

#include <iostream>
#include <string>

static bool parseArgs(int argc, char** argv, RenderOptions& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto requireValue = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--width") {
            const char* value = requireValue("--width");
            if (!value) return false;
            cfg.width = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--height") {
            const char* value = requireValue("--height");
            if (!value) return false;
            cfg.height = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--output") {
            const char* value = requireValue("--output");
            if (!value) return false;
            cfg.output = value;
        } else if (arg == "--scene") {
            const char* value = requireValue("--scene");
            if (!value) return false;
            cfg.scene = value;
        } else if (arg == "--print-options-schema") {
            cfg.print_options_schema = true;
        } else if (arg == "--print-scene-registry") {
            cfg.print_scene_registry = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sr_rt_gpu [--scene name] [--width N] [--height N] [--output path] [--print-options-schema] [--print-scene-registry]\n";
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    RenderOptions cfg;
    if (!parseArgs(argc, argv, cfg)) {
        return argc > 1 ? 1 : 0;
    }

    if (cfg.print_options_schema) {
        std::cout << buildOptionsSchemaJson();
        return 0;
    }
    if (cfg.print_scene_registry) {
        std::cout << buildSceneRegistryJson();
        return 0;
    }

    SceneDescription scene;
    try {
        scene = buildScene(cfg.scene);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }

    std::string error_message;
    if (!renderSceneToFile(cfg, scene, &error_message)) {
        std::cerr << error_message << "\n";
        return 1;
    }

    std::cout << "Wrote " << cfg.output << " (" << cfg.width << "x" << cfg.height << ") using scene '" << scene.name << "'\n";
    return 0;
}
