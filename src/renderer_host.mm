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
        if (outBuffer == nil) {
            if (error_message) *error_message = "Failed to allocate Metal output buffer.";
            return false;
        }
        std::memset([outBuffer contents], 0, pixelCount * sizeof(float) * 4);

        id<MTLBuffer> cameraBuffer = [device newBufferWithBytes:&scene.camera length:sizeof(CameraData) options:MTLResourceStorageModeShared];
        id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:scene.spheres.data() length:sizeof(SphereData) * scene.spheres.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> planeBuffer = [device newBufferWithBytes:scene.planes.data() length:sizeof(PlaneData) * scene.planes.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> triangleBuffer = [device newBufferWithBytes:scene.triangles.data() length:sizeof(TriangleData) * scene.triangles.size() options:MTLResourceStorageModeShared];
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
                simd_make_float3(0.0f, 0.0f, 0.0f),
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
            [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        const auto* accum = static_cast<const float*>([outBuffer contents]);
        std::vector<uint8_t> rgb(pixelCount * 3);
        const float invSamples = 1.0f / static_cast<float>(spp);
        for (size_t i = 0; i < pixelCount; ++i) {
            float r = accum[i * 4 + 0] * invSamples;
            float g = accum[i * 4 + 1] * invSamples;
            float b = accum[i * 4 + 2] * invSamples;

            r = r / (1.0f + r);
            g = g / (1.0f + g);
            b = b / (1.0f + b);

            r = std::sqrt(std::max(0.0f, r));
            g = std::sqrt(std::max(0.0f, g));
            b = std::sqrt(std::max(0.0f, b));

            r = std::clamp(r, 0.0f, 1.0f);
            g = std::clamp(g, 0.0f, 1.0f);
            b = std::clamp(b, 0.0f, 1.0f);

            rgb[i * 3 + 0] = static_cast<uint8_t>(r * 255.0f);
            rgb[i * 3 + 1] = static_cast<uint8_t>(g * 255.0f);
            rgb[i * 3 + 2] = static_cast<uint8_t>(b * 255.0f);
        }

        if (!writePPM(options.output, options.width, options.height, rgb)) {
            if (error_message) *error_message = "Failed to write output image.";
            return false;
        }
        return true;
    }
}
