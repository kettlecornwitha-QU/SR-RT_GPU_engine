#pragma once

#include "gpu_types.hpp"

#include <string>
#include <vector>

SceneDescription buildScene(const std::string& scene_name);
std::vector<std::string> availableSceneNames();
std::string buildOptionsSchemaJson();
std::string buildSceneRegistryJson();
