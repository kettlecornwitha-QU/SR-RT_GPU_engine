#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>

#include "gpu_types.hpp"
#include "renderer_host.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct DenoiseUniforms {
    uint32_t width;
    uint32_t height;
    uint32_t step_size;
    float color_sigma;
    float albedo_sigma;
    float normal_sigma;
    float depth_sigma;
    float roughness_sigma;
};

struct GuidePixel {
    simd_float3 normal;
    simd_float3 albedo;
    float depth;
    float roughness;
};

struct ToneMapSettings {
    bool enabled = true;
    std::string mode = "filmic";
    float display_exposure = 0.0f;
    float contrast = 1.0f;
    float saturation = 1.0f;
    float gamma = 2.2f;
};

struct ImageData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<simd_float4> pixels;
};

std::string readTextFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string trim(std::string_view value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return std::string(value.substr(start, end - start));
}

bool readNextToken(std::istream& input, std::string& token) {
    token.clear();
    while (input >> token) {
        if (!token.empty() && token[0] == '#') {
            std::string comment;
            std::getline(input, comment);
            continue;
        }
        return true;
    }
    return false;
}

bool loadPPMImage(const fs::path& path, ImageData& image) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::string magic;
    if (!readNextToken(in, magic) || (magic != "P3" && magic != "P6")) return false;

    std::string token;
    if (!readNextToken(in, token)) return false;
    image.width = static_cast<uint32_t>(std::stoul(token));
    if (!readNextToken(in, token)) return false;
    image.height = static_cast<uint32_t>(std::stoul(token));
    if (!readNextToken(in, token)) return false;
    const float maxValue = std::max(1.0f, std::stof(token));

    image.pixels.resize(static_cast<size_t>(image.width) * image.height);
    if (magic == "P3") {
        for (size_t i = 0; i < image.pixels.size(); ++i) {
            std::string rs;
            std::string gs;
            std::string bs;
            if (!readNextToken(in, rs) || !readNextToken(in, gs) || !readNextToken(in, bs)) return false;
            image.pixels[i] = simd_make_float4(std::stof(rs) / maxValue,
                                               std::stof(gs) / maxValue,
                                               std::stof(bs) / maxValue,
                                               1.0f);
        }
        return true;
    }

    in.get();
    std::vector<uint8_t> rgb(static_cast<size_t>(image.width) * image.height * 3);
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (in.gcount() != static_cast<std::streamsize>(rgb.size())) return false;
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        image.pixels[i] = simd_make_float4(float(rgb[i * 3 + 0]) / maxValue,
                                           float(rgb[i * 3 + 1]) / maxValue,
                                           float(rgb[i * 3 + 2]) / maxValue,
                                           1.0f);
    }
    return true;
}

bool parseHdrResolution(const std::string& line, uint32_t& width, uint32_t& height) {
    char axisY = 0;
    char signY = 0;
    char axisX = 0;
    char signX = 0;
    int h = 0;
    int w = 0;
    if (std::sscanf(line.c_str(), " %c%c %d %c%c %d", &signY, &axisY, &h, &signX, &axisX, &w) != 6) {
        return false;
    }
    if (axisY != 'Y' || axisX != 'X' || h <= 0 || w <= 0) {
        return false;
    }
    height = static_cast<uint32_t>(h);
    width = static_cast<uint32_t>(w);
    return true;
}

simd_float3 rgbeToFloat(uint8_t r, uint8_t g, uint8_t b, uint8_t e) {
    if (e == 0) return simd_make_float3(0.0f, 0.0f, 0.0f);
    const float scale = std::ldexp(1.0f, int(e) - (128 + 8));
    return simd_make_float3((float(r) + 0.5f) * scale,
                            (float(g) + 0.5f) * scale,
                            (float(b) + 0.5f) * scale);
}

bool loadHDRImage(const fs::path& path, ImageData& image) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    std::string line;
    if (!std::getline(in, line) || line.rfind("#?RADIANCE", 0) != 0) return false;

    bool foundFormat = false;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) break;
        if (line.rfind("FORMAT=", 0) == 0 && line.find("32-bit_rle_rgbe") != std::string::npos) {
            foundFormat = true;
        }
    }
    if (!foundFormat) return false;
    if (!std::getline(in, line)) return false;

    uint32_t width = 0;
    uint32_t height = 0;
    if (!parseHdrResolution(line, width, height)) return false;

    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * height);
    std::vector<uint8_t> scanline(static_cast<size_t>(width) * 4);

    for (uint32_t y = 0; y < height; ++y) {
        uint8_t header[4] = {0, 0, 0, 0};
        in.read(reinterpret_cast<char*>(header), 4);
        if (!in) return false;

        if (width >= 8 && width <= 0x7fff && header[0] == 2 && header[1] == 2 &&
            static_cast<uint32_t>((header[2] << 8) | header[3]) == width) {
            for (int channel = 0; channel < 4; ++channel) {
                size_t x = 0;
                while (x < width) {
                    uint8_t count = 0;
                    uint8_t value = 0;
                    in.read(reinterpret_cast<char*>(&count), 1);
                    in.read(reinterpret_cast<char*>(&value), 1);
                    if (!in) return false;
                    if (count > 128) {
                        size_t runLength = size_t(count - 128);
                        for (size_t i = 0; i < runLength && x < width; ++i, ++x) {
                            scanline[x * 4 + channel] = value;
                        }
                    } else {
                        scanline[x * 4 + channel] = value;
                        ++x;
                        for (size_t i = 1; i < count && x < width; ++i, ++x) {
                            in.read(reinterpret_cast<char*>(&scanline[x * 4 + channel]), 1);
                            if (!in) return false;
                        }
                    }
                }
            }
        } else {
            scanline[0] = header[0];
            scanline[1] = header[1];
            scanline[2] = header[2];
            scanline[3] = header[3];
            in.read(reinterpret_cast<char*>(scanline.data() + 4), static_cast<std::streamsize>(scanline.size() - 4));
            if (!in) return false;
        }

        for (uint32_t x = 0; x < width; ++x) {
            simd_float3 color = rgbeToFloat(scanline[x * 4 + 0],
                                            scanline[x * 4 + 1],
                                            scanline[x * 4 + 2],
                                            scanline[x * 4 + 3]);
            image.pixels[static_cast<size_t>(y) * width + x] = simd_make_float4(color.x, color.y, color.z, 1.0f);
        }
    }
    return true;
}

bool loadEnvironmentImage(const fs::path& path, ImageData& image) {
    const std::string ext = path.extension().string();
    if (ext == ".hdr" || ext == ".HDR") {
        return loadHDRImage(path, image);
    }
    if (ext == ".ppm" || ext == ".PPM") {
        return loadPPMImage(path, image);
    }
    return false;
}

id<MTLTexture> makeEnvironmentTexture(id<MTLDevice> device, const ImageData& image) {
    if (image.width == 0 || image.height == 0 || image.pixels.empty()) return nil;
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                          width:image.width
                                                                                         height:image.height
                                                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (texture == nil) return nil;

    MTLRegion region = MTLRegionMake2D(0, 0, image.width, image.height);
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:image.pixels.data()
               bytesPerRow:image.width * sizeof(simd_float4)];
    return texture;
}

ImageData makeFallbackEnvironment() {
    ImageData image;
    image.width = 1;
    image.height = 1;
    image.pixels = {simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f)};
    return image;
}

bool writePPM(const fs::path& outputPath, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgb) {
    std::error_code ec;
    fs::create_directories(outputPath.parent_path(), ec);
    std::ofstream out(outputPath, std::ios::binary);
    if (!out) return false;
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(out);
}

simd_float3 applySaturation(simd_float3 color, float saturation) {
    const float luma = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
    const simd_float3 gray = simd_make_float3(luma, luma, luma);
    return simd_mix(gray, color, saturation);
}

simd_float3 applyContrast(simd_float3 color, float contrast) {
    return simd_make_float3(0.5f, 0.5f, 0.5f) + (color - simd_make_float3(0.5f, 0.5f, 0.5f)) * contrast;
}

simd_float3 filmicToneMap(simd_float3 color) {
    auto mapChannel = [](float x) {
        const float a = 2.51f;
        const float b = 0.03f;
        const float c = 2.43f;
        const float d = 0.59f;
        const float e = 0.14f;
        return std::clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    };
    return simd_make_float3(mapChannel(color.x), mapChannel(color.y), mapChannel(color.z));
}

simd_float3 reinhardToneMap(simd_float3 color) {
    return color / (simd_make_float3(1.0f, 1.0f, 1.0f) + color);
}

simd_float3 toneMapColor(simd_float3 color, const ToneMapSettings& settings) {
    if (!settings.enabled) {
        return simd_clamp(color, simd_make_float3(0.0f, 0.0f, 0.0f), simd_make_float3(1.0f, 1.0f, 1.0f));
    }

    color *= std::exp2(settings.display_exposure);
    if (settings.mode == "linear") {
        color = simd_clamp(color, simd_make_float3(0.0f, 0.0f, 0.0f), simd_make_float3(1.0f, 1.0f, 1.0f));
    } else if (settings.mode == "reinhard") {
        color = reinhardToneMap(simd_max(color, simd_make_float3(0.0f, 0.0f, 0.0f)));
    } else {
        color = filmicToneMap(simd_max(color, simd_make_float3(0.0f, 0.0f, 0.0f)));
    }

    color = applyContrast(color, settings.contrast);
    color = applySaturation(color, settings.saturation);
    const float invGamma = 1.0f / std::max(0.1f, settings.gamma);
    color = simd_make_float3(std::pow(std::max(0.0f, color.x), invGamma),
                             std::pow(std::max(0.0f, color.y), invGamma),
                             std::pow(std::max(0.0f, color.z), invGamma));
    return simd_clamp(color, simd_make_float3(0.0f, 0.0f, 0.0f), simd_make_float3(1.0f, 1.0f, 1.0f));
}

std::vector<uint8_t> encodeRGB(const std::vector<simd_float3>& image, const ToneMapSettings& settings) {
    std::vector<uint8_t> rgb(image.size() * 3);
    for (size_t i = 0; i < image.size(); ++i) {
        simd_float3 mapped = toneMapColor(image[i], settings);
        rgb[i * 3 + 0] = static_cast<uint8_t>(mapped.x * 255.0f);
        rgb[i * 3 + 1] = static_cast<uint8_t>(mapped.y * 255.0f);
        rgb[i * 3 + 2] = static_cast<uint8_t>(mapped.z * 255.0f);
    }
    return rgb;
}

fs::path sidecarPathFor(const fs::path& outputPath, const std::string& suffix) {
    fs::path stem = outputPath;
    stem.replace_extension();
    return stem.string() + suffix + outputPath.extension().string();
}

fs::path shaderPathForBinary() {
    char pathbuf[4096];
    uint32_t size = sizeof(pathbuf);
    if (_NSGetExecutablePath(pathbuf, &size) != 0) {
        return fs::current_path() / "render_sphere.metal";
    }
    std::error_code ec;
    fs::path exe = fs::weakly_canonical(fs::path(pathbuf), ec);
    if (ec) exe = fs::path(pathbuf);
    return exe.parent_path() / "render_sphere.metal";
}

template <typename T>
id<MTLBuffer> makeSceneBuffer(id<MTLDevice> device, const std::vector<T>& values) {
    if (values.empty()) {
        return [device newBufferWithLength:sizeof(T) options:MTLResourceStorageModeShared];
    }
    return [device newBufferWithBytes:values.data()
                               length:sizeof(T) * values.size()
                              options:MTLResourceStorageModeShared];
}
}

bool renderSceneToFile(const RenderOptions& options, const SceneDescription& scene, std::string* error_message) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            if (error_message) *error_message = "Metal device unavailable.";
            return false;
        }

        std::string shaderText = readTextFile(shaderPathForBinary());
        if (shaderText.empty()) {
            if (error_message) *error_message = "Failed to read shader source.";
            return false;
        }

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:shaderText.c_str()];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            if (error_message) *error_message = std::string("Failed to compile Metal shader: ") + (error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"renderScene"];
        if (function == nil) {
            if (error_message) *error_message = "Failed to find renderScene kernel.";
            return false;
        }
        id<MTLFunction> denoiseFunction = [library newFunctionWithName:@"atrousDenoise"];
        if (denoiseFunction == nil) {
            if (error_message) *error_message = "Failed to find atrousDenoise kernel.";
            return false;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            if (error_message) *error_message = std::string("Failed to create compute pipeline: ") + (error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }
        id<MTLComputePipelineState> denoisePipeline = [device newComputePipelineStateWithFunction:denoiseFunction error:&error];
        if (denoisePipeline == nil) {
            if (error_message) *error_message = std::string("Failed to create denoise pipeline: ") + (error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }

        ImageData environmentImage = makeFallbackEnvironment();
        bool useEnvironmentMap = false;
        if (!options.environment_map.empty()) {
            fs::path environmentPath = fs::path(options.environment_map);
            if (!environmentPath.is_absolute()) {
                environmentPath = fs::current_path() / environmentPath;
            }
            if (!loadEnvironmentImage(environmentPath, environmentImage)) {
                if (error_message) *error_message = "Failed to load environment map. Supported formats: .hdr, .ppm";
                return false;
            }
            useEnvironmentMap = true;
        }
        id<MTLTexture> environmentTexture = makeEnvironmentTexture(device, environmentImage);
        if (environmentTexture == nil) {
            if (error_message) *error_message = "Failed to create Metal environment texture.";
            return false;
        }

        const size_t pixelCount = static_cast<size_t>(options.width) * static_cast<size_t>(options.height);
        id<MTLBuffer> outBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> normalBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> albedoBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> depthRoughnessBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> diffuseBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> specularBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> transmissionBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> momentBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> activeMaskBuffer = [device newBufferWithLength:pixelCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (outBuffer == nil || normalBuffer == nil || albedoBuffer == nil || depthRoughnessBuffer == nil ||
            diffuseBuffer == nil || specularBuffer == nil || transmissionBuffer == nil ||
            momentBuffer == nil || activeMaskBuffer == nil) {
            if (error_message) *error_message = "Failed to allocate Metal output buffers.";
            return false;
        }
        std::memset([outBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([normalBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([albedoBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([depthRoughnessBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([diffuseBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([specularBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([transmissionBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([momentBuffer contents], 0, pixelCount * sizeof(float) * 4);
        auto* activeMask = static_cast<uint32_t*>([activeMaskBuffer contents]);
        for (size_t i = 0; i < pixelCount; ++i) {
            activeMask[i] = 1u;
        }

        id<MTLBuffer> cameraBuffer = [device newBufferWithBytes:&scene.camera length:sizeof(CameraData) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereBuffer = makeSceneBuffer(device, scene.spheres);
        id<MTLBuffer> planeBuffer = makeSceneBuffer(device, scene.planes);
        id<MTLBuffer> triangleBuffer = makeSceneBuffer(device, scene.triangles);
        if (cameraBuffer == nil || sphereBuffer == nil || planeBuffer == nil || triangleBuffer == nil) {
            if (error_message) *error_message = "Failed to allocate Metal scene buffers.";
            return false;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            if (error_message) *error_message = "Failed to create command queue.";
            return false;
        }

        NSUInteger threadWidth = pipeline.threadExecutionWidth;
        NSUInteger threadHeight = pipeline.maxTotalThreadsPerThreadgroup / threadWidth;
        if (threadHeight == 0) threadHeight = 1;
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
        MTLSize threadsPerGrid = MTLSizeMake(options.width, options.height, 1);

        const uint32_t spp = std::max<uint32_t>(1u, options.spp);
        const uint32_t adaptiveMinSpp = std::min(std::max<uint32_t>(1u, options.adaptive_min_spp), spp);
        const uint32_t adaptiveCheckInterval = 4u;
        size_t activePixelCount = pixelCount;
        for (uint32_t sample = 0; sample < spp; ++sample) {
            RenderUniforms uniforms {
                options.width,
                options.height,
                static_cast<uint32_t>(scene.spheres.size()),
                static_cast<uint32_t>(scene.planes.size()),
                static_cast<uint32_t>(scene.triangles.size()),
                spp,
                sample,
                useEnvironmentMap ? 1u : 0u,
                0.0f,
                0.035f,
                options.firefly_clamp,
                options.environment_rotation * (3.14159265358979323846f / 180.0f),
                std::exp2(options.environment_exposure),
                simd_make_float2(0.0f, 0.0f),
            };

            id<MTLBuffer> uniformBuffer = [device newBufferWithBytes:&uniforms length:sizeof(RenderUniforms) options:MTLResourceStorageModeShared];
            if (uniformBuffer == nil) {
                if (error_message) *error_message = "Failed to allocate Metal uniform buffer.";
                return false;
            }

            id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:outBuffer offset:0 atIndex:0];
            [encoder setBuffer:uniformBuffer offset:0 atIndex:1];
            [encoder setBuffer:cameraBuffer offset:0 atIndex:2];
            [encoder setBuffer:sphereBuffer offset:0 atIndex:3];
            [encoder setBuffer:planeBuffer offset:0 atIndex:4];
            [encoder setBuffer:triangleBuffer offset:0 atIndex:5];
            [encoder setBuffer:normalBuffer offset:0 atIndex:6];
            [encoder setBuffer:albedoBuffer offset:0 atIndex:7];
            [encoder setBuffer:depthRoughnessBuffer offset:0 atIndex:8];
            [encoder setBuffer:diffuseBuffer offset:0 atIndex:9];
            [encoder setBuffer:specularBuffer offset:0 atIndex:10];
            [encoder setBuffer:transmissionBuffer offset:0 atIndex:11];
            [encoder setBuffer:momentBuffer offset:0 atIndex:12];
            [encoder setBuffer:activeMaskBuffer offset:0 atIndex:13];
            [encoder setTexture:environmentTexture atIndex:0];
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            if (options.adaptive_sampling &&
                sample + 1 >= adaptiveMinSpp &&
                (((sample + 1) % adaptiveCheckInterval) == 0u || (sample + 1) == spp)) {
                const auto* accumSnapshot = static_cast<const float*>([outBuffer contents]);
                const auto* momentSnapshot = static_cast<const float*>([momentBuffer contents]);
                activePixelCount = 0;
                for (size_t i = 0; i < pixelCount; ++i) {
                    if (activeMask[i] == 0u) {
                        continue;
                    }
                    float sampleCount = accumSnapshot[i * 4 + 3];
                    if (sampleCount < static_cast<float>(adaptiveMinSpp)) {
                        ++activePixelCount;
                        continue;
                    }

                    simd_float3 mean = simd_make_float3(accumSnapshot[i * 4 + 0],
                                                        accumSnapshot[i * 4 + 1],
                                                        accumSnapshot[i * 4 + 2]) / sampleCount;
                    simd_float3 meanSq = simd_make_float3(momentSnapshot[i * 4 + 0],
                                                          momentSnapshot[i * 4 + 1],
                                                          momentSnapshot[i * 4 + 2]) / sampleCount;
                    simd_float3 variance3 = simd_max(meanSq - mean * mean, simd_make_float3(0.0f, 0.0f, 0.0f));
                    float variance = std::max(variance3.x, std::max(variance3.y, variance3.z));
                    float meanLuma = std::max(0.05f, 0.2126f * mean.x + 0.7152f * mean.y + 0.0722f * mean.z);
                    float relativeError = std::sqrt(variance) / meanLuma;
                    if (relativeError < options.adaptive_threshold) {
                        activeMask[i] = 0u;
                    } else {
                        ++activePixelCount;
                    }
                }
                if (activePixelCount == 0) {
                    break;
                }
            }
        }

        const auto* accum = static_cast<const float*>([outBuffer contents]);
        const auto* normalAccum = static_cast<const float*>([normalBuffer contents]);
        const auto* albedoAccum = static_cast<const float*>([albedoBuffer contents]);
        const auto* depthRoughnessAccum = static_cast<const float*>([depthRoughnessBuffer contents]);
        const auto* diffuseAccum = static_cast<const float*>([diffuseBuffer contents]);
        const auto* specularAccum = static_cast<const float*>([specularBuffer contents]);
        const auto* transmissionAccum = static_cast<const float*>([transmissionBuffer contents]);
        std::vector<simd_float3> beauty(pixelCount);
        std::vector<simd_float3> diffuseImage(pixelCount);
        std::vector<simd_float3> specularImage(pixelCount);
        std::vector<simd_float3> transmissionImage(pixelCount);
        std::vector<float> sampleCountImage(pixelCount);
        std::vector<GuidePixel> guides(pixelCount);
        id<MTLBuffer> componentBufferA = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> componentBufferB = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> guideNormalBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> guideAlbedoBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> guideDepthRoughnessBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        if (componentBufferA == nil || componentBufferB == nil || guideNormalBuffer == nil || guideAlbedoBuffer == nil || guideDepthRoughnessBuffer == nil) {
            if (error_message) *error_message = "Failed to allocate denoise buffers.";
            return false;
        }
        float minDepth = std::numeric_limits<float>::max();
        float maxDepth = 0.0f;
        auto* guideNormal = static_cast<float*>([guideNormalBuffer contents]);
        auto* guideAlbedo = static_cast<float*>([guideAlbedoBuffer contents]);
        auto* guideDepthRoughness = static_cast<float*>([guideDepthRoughnessBuffer contents]);
        for (size_t i = 0; i < pixelCount; ++i) {
            float sampleCount = std::max(1.0f, accum[i * 4 + 3]);
            float invCount = 1.0f / sampleCount;
            sampleCountImage[i] = sampleCount;
            beauty[i] = simd_make_float3(accum[i * 4 + 0], accum[i * 4 + 1], accum[i * 4 + 2]) * invCount;
            diffuseImage[i] = simd_make_float3(diffuseAccum[i * 4 + 0], diffuseAccum[i * 4 + 1], diffuseAccum[i * 4 + 2]) * invCount;
            specularImage[i] = simd_make_float3(specularAccum[i * 4 + 0], specularAccum[i * 4 + 1], specularAccum[i * 4 + 2]) * invCount;
            transmissionImage[i] = simd_make_float3(transmissionAccum[i * 4 + 0], transmissionAccum[i * 4 + 1], transmissionAccum[i * 4 + 2]) * invCount;
            simd_float3 normal = simd_make_float3(normalAccum[i * 4 + 0], normalAccum[i * 4 + 1], normalAccum[i * 4 + 2]) * invCount;
            simd_float3 albedo = simd_make_float3(albedoAccum[i * 4 + 0], albedoAccum[i * 4 + 1], albedoAccum[i * 4 + 2]) * invCount;
            float depth = depthRoughnessAccum[i * 4 + 0] * invCount;
            float roughness = depthRoughnessAccum[i * 4 + 1] * invCount;
            guides[i].normal = simd_normalize(simd_max(normal * 2.0f - 1.0f, simd_make_float3(-1.0f, -1.0f, -1.0f)));
            guides[i].albedo = albedo;
            guides[i].depth = depth;
            guides[i].roughness = roughness;
            guideNormal[i * 4 + 0] = normal.x;
            guideNormal[i * 4 + 1] = normal.y;
            guideNormal[i * 4 + 2] = normal.z;
            guideNormal[i * 4 + 3] = 1.0f;
            guideAlbedo[i * 4 + 0] = albedo.x;
            guideAlbedo[i * 4 + 1] = albedo.y;
            guideAlbedo[i * 4 + 2] = albedo.z;
            guideAlbedo[i * 4 + 3] = 1.0f;
            guideDepthRoughness[i * 4 + 0] = depth;
            guideDepthRoughness[i * 4 + 1] = roughness;
            guideDepthRoughness[i * 4 + 2] = 0.0f;
            guideDepthRoughness[i * 4 + 3] = 1.0f;
            if (depth > 0.0f) {
                minDepth = std::min(minDepth, depth);
                maxDepth = std::max(maxDepth, depth);
            }
        }

        std::vector<simd_float3> finalBeauty = beauty;
        const ToneMapSettings toneSettings {
            true,
            options.tonemap,
            options.display_exposure,
            options.tone_contrast,
            options.tone_saturation,
            options.output_gamma,
        };
        const ToneMapSettings rawSettings {
            false,
            "linear",
            0.0f,
            1.0f,
            1.0f,
            1.0f,
        };
        if (options.denoise && options.denoise_strength > 0.0f) {
            const float baseColorSigma = std::max(0.08f, 0.45f * options.denoise_strength);
            const float baseAlbedoSigma = std::max(0.06f, 0.30f * options.denoise_strength);
            const float baseNormalSigma = std::max(0.04f, 0.18f * options.denoise_strength);
            const float baseDepthSigma = std::max(0.03f, 0.14f * options.denoise_strength);
            const float baseRoughnessSigma = std::max(0.02f, 0.12f * options.denoise_strength);

            auto denoiseComponent = [&](std::vector<simd_float3>& image,
                                        float colorScale,
                                        float albedoScale,
                                        float normalScale,
                                        float depthScale,
                                        float roughnessScale,
                                        int passCount) -> bool {
                auto* componentA = static_cast<float*>([componentBufferA contents]);
                for (size_t i = 0; i < pixelCount; ++i) {
                    componentA[i * 4 + 0] = image[i].x;
                    componentA[i * 4 + 1] = image[i].y;
                    componentA[i * 4 + 2] = image[i].z;
                    componentA[i * 4 + 3] = 1.0f;
                }

                id<MTLBuffer> currentSrc = componentBufferA;
                id<MTLBuffer> currentDst = componentBufferB;
                for (int pass = 0; pass < passCount; ++pass) {
                    DenoiseUniforms denoiseUniforms {
                        options.width,
                        options.height,
                        static_cast<uint32_t>(1u << pass),
                        baseColorSigma * colorScale,
                        baseAlbedoSigma * albedoScale,
                        baseNormalSigma * normalScale,
                        baseDepthSigma * depthScale,
                        baseRoughnessSigma * roughnessScale,
                    };
                    id<MTLBuffer> denoiseUniformBuffer = [device newBufferWithBytes:&denoiseUniforms length:sizeof(DenoiseUniforms) options:MTLResourceStorageModeShared];
                    if (denoiseUniformBuffer == nil) {
                        if (error_message) *error_message = "Failed to allocate denoise uniform buffer.";
                        return false;
                    }

                    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    [encoder setComputePipelineState:denoisePipeline];
                    [encoder setBuffer:currentSrc offset:0 atIndex:0];
                    [encoder setBuffer:currentDst offset:0 atIndex:1];
                    [encoder setBuffer:guideNormalBuffer offset:0 atIndex:2];
                    [encoder setBuffer:guideAlbedoBuffer offset:0 atIndex:3];
                    [encoder setBuffer:guideDepthRoughnessBuffer offset:0 atIndex:4];
                    [encoder setBuffer:denoiseUniformBuffer offset:0 atIndex:5];
                    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                    [encoder endEncoding];
                    [commandBuffer commit];
                    [commandBuffer waitUntilCompleted];
                    std::swap(currentSrc, currentDst);
                }

                const auto* denoised = static_cast<const float*>([currentSrc contents]);
                for (size_t i = 0; i < pixelCount; ++i) {
                    image[i] = simd_make_float3(denoised[i * 4 + 0], denoised[i * 4 + 1], denoised[i * 4 + 2]);
                }
                return true;
            };

            if (!denoiseComponent(diffuseImage, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 4)) {
                return false;
            }
            if (!denoiseComponent(specularImage, 0.42f, 0.45f, 0.70f, 0.78f, 0.82f, 2)) {
                return false;
            }
            if (!denoiseComponent(transmissionImage, 0.52f, 0.48f, 0.72f, 0.82f, 0.86f, 2)) {
                return false;
            }

            for (size_t i = 0; i < pixelCount; ++i) {
                finalBeauty[i] = diffuseImage[i] + specularImage[i] + transmissionImage[i];
            }
        }

        if (options.save_guide_buffers) {
            std::vector<simd_float3> normalImage(pixelCount);
            std::vector<simd_float3> albedoImage(pixelCount);
            std::vector<simd_float3> depthImage(pixelCount);
            std::vector<simd_float3> roughnessImage(pixelCount);
            std::vector<simd_float3> samplesImage(pixelCount);
            const float depthRange = std::max(maxDepth - minDepth, 1e-4f);
            float minSamples = std::numeric_limits<float>::max();
            float maxSamples = 0.0f;
            for (float count : sampleCountImage) {
                minSamples = std::min(minSamples, count);
                maxSamples = std::max(maxSamples, count);
            }
            float sampleRange = std::max(maxSamples - minSamples, 1e-4f);
            for (size_t i = 0; i < pixelCount; ++i) {
                normalImage[i] = guides[i].normal * 0.5f + simd_make_float3(0.5f, 0.5f, 0.5f);
                albedoImage[i] = guides[i].albedo;
                float normalizedDepth = guides[i].depth > 0.0f ? (guides[i].depth - minDepth) / depthRange : 1.0f;
                depthImage[i] = simd_make_float3(normalizedDepth, normalizedDepth, normalizedDepth);
                roughnessImage[i] = simd_make_float3(guides[i].roughness, guides[i].roughness, guides[i].roughness);
                float normalizedSamples = (sampleCountImage[i] - minSamples) / sampleRange;
                samplesImage[i] = simd_make_float3(normalizedSamples, normalizedSamples, normalizedSamples);
            }

            if (!writePPM(sidecarPathFor(options.output, "_normal"), options.width, options.height, encodeRGB(normalImage, rawSettings))) {
                if (error_message) *error_message = "Failed to write normal guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_albedo"), options.width, options.height, encodeRGB(albedoImage, rawSettings))) {
                if (error_message) *error_message = "Failed to write albedo guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_depth"), options.width, options.height, encodeRGB(depthImage, rawSettings))) {
                if (error_message) *error_message = "Failed to write depth guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_roughness"), options.width, options.height, encodeRGB(roughnessImage, rawSettings))) {
                if (error_message) *error_message = "Failed to write roughness guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_samples"), options.width, options.height, encodeRGB(samplesImage, rawSettings))) {
                if (error_message) *error_message = "Failed to write sample-count image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_diffuse"), options.width, options.height, encodeRGB(diffuseImage, toneSettings))) {
                if (error_message) *error_message = "Failed to write diffuse component image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_specular"), options.width, options.height, encodeRGB(specularImage, toneSettings))) {
                if (error_message) *error_message = "Failed to write specular component image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_transmission"), options.width, options.height, encodeRGB(transmissionImage, toneSettings))) {
                if (error_message) *error_message = "Failed to write transmission component image.";
                return false;
            }
        }

        if (!writePPM(options.output, options.width, options.height, encodeRGB(finalBeauty, toneSettings))) {
            if (error_message) *error_message = "Failed to write output image.";
            return false;
        }
        return true;
    }
}
