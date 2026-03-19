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
        } else if (arg == "--spp") {
            const char* value = requireValue("--spp");
            if (!value) return false;
            cfg.spp = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--adaptive-min-spp") {
            const char* value = requireValue("--adaptive-min-spp");
            if (!value) return false;
            cfg.adaptive_min_spp = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--output") {
            const char* value = requireValue("--output");
            if (!value) return false;
            cfg.output = value;
        } else if (arg == "--tonemap") {
            const char* value = requireValue("--tonemap");
            if (!value) return false;
            cfg.tonemap = value;
        } else if (arg == "--environment-map") {
            const char* value = requireValue("--environment-map");
            if (!value) return false;
            cfg.environment_map = value;
        } else if (arg == "--scene") {
            const char* value = requireValue("--scene");
            if (!value) return false;
            cfg.scene = value;
        } else if (arg == "--firefly-clamp") {
            const char* value = requireValue("--firefly-clamp");
            if (!value) return false;
            cfg.firefly_clamp = std::stof(value);
        } else if (arg == "--denoise-strength") {
            const char* value = requireValue("--denoise-strength");
            if (!value) return false;
            cfg.denoise_strength = std::stof(value);
        } else if (arg == "--display-exposure") {
            const char* value = requireValue("--display-exposure");
            if (!value) return false;
            cfg.display_exposure = std::stof(value);
        } else if (arg == "--tone-contrast") {
            const char* value = requireValue("--tone-contrast");
            if (!value) return false;
            cfg.tone_contrast = std::stof(value);
        } else if (arg == "--tone-saturation") {
            const char* value = requireValue("--tone-saturation");
            if (!value) return false;
            cfg.tone_saturation = std::stof(value);
        } else if (arg == "--output-gamma") {
            const char* value = requireValue("--output-gamma");
            if (!value) return false;
            cfg.output_gamma = std::stof(value);
        } else if (arg == "--adaptive-threshold") {
            const char* value = requireValue("--adaptive-threshold");
            if (!value) return false;
            cfg.adaptive_threshold = std::stof(value);
        } else if (arg == "--environment-rotation") {
            const char* value = requireValue("--environment-rotation");
            if (!value) return false;
            cfg.environment_rotation = std::stof(value);
        } else if (arg == "--environment-exposure") {
            const char* value = requireValue("--environment-exposure");
            if (!value) return false;
            cfg.environment_exposure = std::stof(value);
        } else if (arg == "--adaptive-sampling") {
            cfg.adaptive_sampling = true;
        } else if (arg == "--oidn-denoise") {
            cfg.oidn_denoise = true;
        } else if (arg == "--no-denoise") {
            cfg.denoise = false;
        } else if (arg == "--save-guide-buffers") {
            cfg.save_guide_buffers = true;
        } else if (arg == "--print-options-schema") {
            cfg.print_options_schema = true;
        } else if (arg == "--print-scene-registry") {
            cfg.print_scene_registry = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sr_rt_gpu [--scene name] [--width N] [--height N] [--spp N] [--adaptive-sampling] [--adaptive-min-spp N] [--adaptive-threshold V] [--environment-map path] [--environment-rotation deg] [--environment-exposure V] [--tonemap filmic|reinhard|linear] [--display-exposure V] [--tone-contrast V] [--tone-saturation V] [--output-gamma V] [--output path] [--firefly-clamp V] [--denoise-strength V] [--oidn-denoise] [--no-denoise] [--save-guide-buffers] [--print-options-schema] [--print-scene-registry]\n";
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

    std::cout << "Wrote " << cfg.output << " (" << cfg.width << "x" << cfg.height << ", spp=" << cfg.spp << ") using scene '" << scene.name << "'";
    if (cfg.denoise) {
        std::cout << " with denoise_strength=" << cfg.denoise_strength;
    } else {
        std::cout << " with denoising disabled";
    }
    std::cout << " and firefly_clamp=" << cfg.firefly_clamp << "\n";
    if (cfg.adaptive_sampling) {
        std::cout << "Adaptive sampling enabled with min_spp=" << cfg.adaptive_min_spp
                  << " and threshold=" << cfg.adaptive_threshold << "\n";
    }
    if (cfg.oidn_denoise) {
        std::cout << "OIDN denoise enabled\n";
    }
    if (!cfg.environment_map.empty()) {
        std::cout << "Environment map: " << cfg.environment_map
                  << " rotation=" << cfg.environment_rotation
                  << " exposure=" << cfg.environment_exposure << "\n";
    }
    std::cout << "Tonemap: " << cfg.tonemap
              << " display_exposure=" << cfg.display_exposure
              << " contrast=" << cfg.tone_contrast
              << " saturation=" << cfg.tone_saturation
              << " gamma=" << cfg.output_gamma << "\n";
    return 0;
}
