#pragma once

#include "gpu_types.hpp"

#include <string>

bool renderSceneToFile(const RenderOptions& options, const SceneDescription& scene, std::string* error_message = nullptr);
