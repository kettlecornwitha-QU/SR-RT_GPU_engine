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
#include <limits>
#include <sstream>
#include <string>
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

        const size_t pixelCount = static_cast<size_t>(options.width) * static_cast<size_t>(options.height);
        id<MTLBuffer> outBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> normalBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> albedoBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> depthRoughnessBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> diffuseBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> specularBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> transmissionBuffer = [device newBufferWithLength:pixelCount * sizeof(float) * 4 options:MTLResourceStorageModeShared];
        if (outBuffer == nil || normalBuffer == nil || albedoBuffer == nil || depthRoughnessBuffer == nil ||
            diffuseBuffer == nil || specularBuffer == nil || transmissionBuffer == nil) {
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
            [encoder setBuffer:depthRoughnessBuffer offset:0 atIndex:8];
            [encoder setBuffer:diffuseBuffer offset:0 atIndex:9];
            [encoder setBuffer:specularBuffer offset:0 atIndex:10];
            [encoder setBuffer:transmissionBuffer offset:0 atIndex:11];
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
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
        const float invSamples = 1.0f / static_cast<float>(spp);
        float minDepth = std::numeric_limits<float>::max();
        float maxDepth = 0.0f;
        auto* guideNormal = static_cast<float*>([guideNormalBuffer contents]);
        auto* guideAlbedo = static_cast<float*>([guideAlbedoBuffer contents]);
        auto* guideDepthRoughness = static_cast<float*>([guideDepthRoughnessBuffer contents]);
        for (size_t i = 0; i < pixelCount; ++i) {
            beauty[i] = simd_make_float3(accum[i * 4 + 0], accum[i * 4 + 1], accum[i * 4 + 2]) * invSamples;
            diffuseImage[i] = simd_make_float3(diffuseAccum[i * 4 + 0], diffuseAccum[i * 4 + 1], diffuseAccum[i * 4 + 2]) * invSamples;
            specularImage[i] = simd_make_float3(specularAccum[i * 4 + 0], specularAccum[i * 4 + 1], specularAccum[i * 4 + 2]) * invSamples;
            transmissionImage[i] = simd_make_float3(transmissionAccum[i * 4 + 0], transmissionAccum[i * 4 + 1], transmissionAccum[i * 4 + 2]) * invSamples;
            simd_float3 normal = simd_make_float3(normalAccum[i * 4 + 0], normalAccum[i * 4 + 1], normalAccum[i * 4 + 2]) * invSamples;
            simd_float3 albedo = simd_make_float3(albedoAccum[i * 4 + 0], albedoAccum[i * 4 + 1], albedoAccum[i * 4 + 2]) * invSamples;
            float depth = depthRoughnessAccum[i * 4 + 0] * invSamples;
            float roughness = depthRoughnessAccum[i * 4 + 1] * invSamples;
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
            const float depthRange = std::max(maxDepth - minDepth, 1e-4f);
            for (size_t i = 0; i < pixelCount; ++i) {
                normalImage[i] = guides[i].normal * 0.5f + simd_make_float3(0.5f, 0.5f, 0.5f);
                albedoImage[i] = guides[i].albedo;
                float normalizedDepth = guides[i].depth > 0.0f ? (guides[i].depth - minDepth) / depthRange : 1.0f;
                depthImage[i] = simd_make_float3(normalizedDepth, normalizedDepth, normalizedDepth);
                roughnessImage[i] = simd_make_float3(guides[i].roughness, guides[i].roughness, guides[i].roughness);
            }

            if (!writePPM(sidecarPathFor(options.output, "_normal"), options.width, options.height, encodeRGB(normalImage, false))) {
                if (error_message) *error_message = "Failed to write normal guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_albedo"), options.width, options.height, encodeRGB(albedoImage, false))) {
                if (error_message) *error_message = "Failed to write albedo guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_depth"), options.width, options.height, encodeRGB(depthImage, false))) {
                if (error_message) *error_message = "Failed to write depth guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_roughness"), options.width, options.height, encodeRGB(roughnessImage, false))) {
                if (error_message) *error_message = "Failed to write roughness guide image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_diffuse"), options.width, options.height, encodeRGB(diffuseImage, true))) {
                if (error_message) *error_message = "Failed to write diffuse component image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_specular"), options.width, options.height, encodeRGB(specularImage, true))) {
                if (error_message) *error_message = "Failed to write specular component image.";
                return false;
            }
            if (!writePPM(sidecarPathFor(options.output, "_transmission"), options.width, options.height, encodeRGB(transmissionImage, true))) {
                if (error_message) *error_message = "Failed to write transmission component image.";
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
