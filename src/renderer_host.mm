#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>

#include "gpu_types.hpp"
#include "renderer_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct GuidePixel {
    simd_float3 normal;
    simd_float3 albedo;
};

std::string readTextFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
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

std::vector<uint8_t> encodeRGB(const std::vector<simd_float3>& image, bool tonemap) {
    std::vector<uint8_t> rgb(image.size() * 3);
    for (size_t i = 0; i < image.size(); ++i) {
        float r = image[i].x;
        float g = image[i].y;
        float b = image[i].z;

        if (tonemap) {
            r = r / (1.0f + r);
            g = g / (1.0f + g);
            b = b / (1.0f + b);
            r = std::sqrt(std::max(0.0f, r));
            g = std::sqrt(std::max(0.0f, g));
            b = std::sqrt(std::max(0.0f, b));
        }

        r = std::clamp(r, 0.0f, 1.0f);
        g = std::clamp(g, 0.0f, 1.0f);
        b = std::clamp(b, 0.0f, 1.0f);
        rgb[i * 3 + 0] = static_cast<uint8_t>(r * 255.0f);
        rgb[i * 3 + 1] = static_cast<uint8_t>(g * 255.0f);
        rgb[i * 3 + 2] = static_cast<uint8_t>(b * 255.0f);
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

std::vector<simd_float3> denoiseBeauty(const std::vector<simd_float3>& beauty,
                                       const std::vector<GuidePixel>& guides,
                                       uint32_t width,
                                       uint32_t height,
                                       float strength) {
    if (strength <= 0.0f) {
        return beauty;
    }
    std::vector<simd_float3> filtered(beauty.size(), simd_make_float3(0.0f, 0.0f, 0.0f));
    const int radius = 2;
    const float spatialSigma = 1.6f;
    const float colorSigma = std::max(0.08f, 0.45f * strength);
    const float albedoSigma = std::max(0.06f, 0.30f * strength);
    const float normalSigma = std::max(0.04f, 0.18f * strength);

    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t idx = static_cast<size_t>(y) * width + x;
            const simd_float3 centerColor = beauty[idx];
            const simd_float3 centerAlbedo = guides[idx].albedo;
            const simd_float3 centerNormal = guides[idx].normal;
            simd_float3 accum = simd_make_float3(0.0f, 0.0f, 0.0f);
            float weightSum = 0.0f;

            for (int oy = -radius; oy <= radius; ++oy) {
                int sy = static_cast<int>(y) + oy;
                if (sy < 0 || sy >= static_cast<int>(height)) continue;
                for (int ox = -radius; ox <= radius; ++ox) {
                    int sx = static_cast<int>(x) + ox;
                    if (sx < 0 || sx >= static_cast<int>(width)) continue;

                    const size_t sampleIdx = static_cast<size_t>(sy) * width + sx;
                    const simd_float3 sampleColor = beauty[sampleIdx];
                    const simd_float3 sampleAlbedo = guides[sampleIdx].albedo;
                    const simd_float3 sampleNormal = guides[sampleIdx].normal;

                    const float spatial2 = static_cast<float>(ox * ox + oy * oy);
                    const float colorDist = simd_length(sampleColor - centerColor);
                    const float albedoDist = simd_length(sampleAlbedo - centerAlbedo);
                    const float normalDist = 1.0f - std::clamp(simd_dot(sampleNormal, centerNormal), 0.0f, 1.0f);

                    const float spatialWeight = std::exp(-spatial2 / (2.0f * spatialSigma * spatialSigma));
                    const float colorWeight = std::exp(-(colorDist * colorDist) / (2.0f * colorSigma * colorSigma));
                    const float albedoWeight = std::exp(-(albedoDist * albedoDist) / (2.0f * albedoSigma * albedoSigma));
                    const float normalWeight = std::exp(-(normalDist * normalDist) / (2.0f * normalSigma * normalSigma));
                    const float weight = spatialWeight * colorWeight * albedoWeight * normalWeight;

                    accum += sampleColor * weight;
                    weightSum += weight;
                }
            }

            filtered[idx] = weightSum > 0.0f ? accum / weightSum : centerColor;
        }
    }

    return filtered;
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

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            if (error_message) *error_message = std::string("Failed to create compute pipeline: ") + (error ? [[error localizedDescription] UTF8String] : "unknown error");
            return false;
        }

        const size_t pixelCount = static_cast<size_t>(options.width) * static_cast<size_t>(options.height);
        id<MTLBuffer> outBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> normalBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> albedoBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        if (outBuffer == nil || normalBuffer == nil || albedoBuffer == nil) {
            if (error_message) *error_message = "Failed to allocate Metal output buffers.";
            return false;
        }
        std::memset([outBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([normalBuffer contents], 0, pixelCount * sizeof(float) * 4);
        std::memset([albedoBuffer contents], 0, pixelCount * sizeof(float) * 4);

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
        for (uint32_t sample = 0; sample < spp; ++sample) {
            RenderUniforms uniforms {
                options.width,
                options.height,
                static_cast<uint32_t>(scene.spheres.size()),
                static_cast<uint32_t>(scene.planes.size()),
                static_cast<uint32_t>(scene.triangles.size()),
                spp,
                sample,
                0.0f,
                0.035f,
                options.firefly_clamp,
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
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        const auto* accum = static_cast<const float*>([outBuffer contents]);
        const auto* normalAccum = static_cast<const float*>([normalBuffer contents]);
        const auto* albedoAccum = static_cast<const float*>([albedoBuffer contents]);
        std::vector<simd_float3> beauty(pixelCount);
        std::vector<GuidePixel> guides(pixelCount);
        const float invSamples = 1.0f / static_cast<float>(spp);
        for (size_t i = 0; i < pixelCount; ++i) {
            beauty[i] = simd_make_float3(accum[i * 4 + 0], accum[i * 4 + 1], accum[i * 4 + 2]) * invSamples;
            simd_float3 normal = simd_make_float3(normalAccum[i * 4 + 0], normalAccum[i * 4 + 1], normalAccum[i * 4 + 2]) * invSamples;
            simd_float3 albedo = simd_make_float3(albedoAccum[i * 4 + 0], albedoAccum[i * 4 + 1], albedoAccum[i * 4 + 2]) * invSamples;
            guides[i].normal = simd_normalize(simd_max(normal * 2.0f - 1.0f, simd_make_float3(-1.0f, -1.0f, -1.0f)));
            guides[i].albedo = albedo;
        }

        std::vector<simd_float3> finalBeauty = options.denoise
            ? denoiseBeauty(beauty, guides, options.width, options.height, options.denoise_strength)
            : beauty;

        if (options.save_guide_buffers) {
            std::vector<simd_float3> normalImage(pixelCount);
            std::vector<simd_float3> albedoImage(pixelCount);
            for (size_t i = 0; i < pixelCount; ++i) {
                normalImage[i] = guides[i].normal * 0.5f + simd_make_float3(0.5f, 0.5f, 0.5f);
                albedoImage[i] = guides[i].albedo;
            }

            if (!writePPM(sidecarPathFor(options.output, "_normal"), options.width, options.height, encodeRGB(normalImage, false))) {
                if (error_message) *error_message = "Failed to write normal guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_albedo"), options.width, options.height, encodeRGB(albedoImage, false))) {
                if (error_message) *error_message = "Failed to write albedo guide image.";
                return false;
            }
        }

        if (!writePPM(options.output, options.width, options.height, encodeRGB(finalBeauty, true))) {
            if (error_message) *error_message = "Failed to write output image.";
            return false;
        }
        return true;
    }
}
