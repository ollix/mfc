// Copyright (c) 2019 Ollix. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ---
// Author: olliwang@ollix.com (Olli Wang)

#import "mfc/MFCGaussianBlurFilter.h"

#include <string>

#import <Metal/Metal.h>

@implementation MFCGaussianBlurFilter

- (id)initWithDevice:(id<MTLDevice>)device
          blurRadius:(float)blurRadius
            andSigma:(float)sigma {
  if (self = [super initWithDevice:device]) {
    _blurRadius = blurRadius;
    _sigma = sigma;
  }
  return self;
}

- (void)applyFilter {
  typedef struct {
    float texelWidthOffset;
    float texelHeightOffset;
  } Uniforms;

  id<MTLBuffer> uniformBuffer = [self.device
      newBufferWithLength:sizeof(Uniforms)
      options:MTLResourceStorageModeShared];
  Uniforms* uniforms = reinterpret_cast<Uniforms*>([uniformBuffer contents]);

  // First pass. Applies Gaussian blur to the input texture for horizontal
  // direction.
  uniforms->texelWidthOffset = 1.0 / self.viewPortSize.x;
  uniforms->texelHeightOffset = 0;
  [self renderToTexture:self.tempTexture
      withSourceTexture:self.texture
      vertexFunctionName:@"vertexShader"
      fragmentFunctionName:@"fragmentShader"
      sourceBlendFactor:MTLBlendFactorOne
      destinationBlendFactor:MTLBlendFactorOneMinusSourceAlpha
      uniformBuffer:uniformBuffer];

  // Second pass. Applies Gaussian blur to the `framebuffer`'s internal texture
  // for vertical direction.
  uniforms->texelWidthOffset = 0;
  uniforms->texelHeightOffset = 1.0 / self.viewPortSize.y;
  [self renderToTexture:self.texture
      withSourceTexture:self.tempTexture
      vertexFunctionName:@"vertexShader"
      fragmentFunctionName:@"fragmentShader"
      sourceBlendFactor:MTLBlendFactorOne
      destinationBlendFactor:MTLBlendFactorOneMinusSourceAlpha
      uniformBuffer:uniformBuffer];
}

- (std::string)shader {
  const int kBlurRadius = std::round(_blurRadius * self.devicePixelRatio);
  if (kBlurRadius <= 0) return "";
  const float kSigma = _sigma * self.devicePixelRatio;

  // First, generate the normal Gaussian weights for a given `sigma_`.
  const int kNumberOfStandardGaussianWeights = kBlurRadius + 2;
  float* standard_gaussian_weights = reinterpret_cast<float*>(
      std::calloc(kNumberOfStandardGaussianWeights, sizeof(float)));
  standard_gaussian_weights[kNumberOfStandardGaussianWeights - 1] = 0;
  float sum_of_weights = 0.0;
  for (int index = 0; index < kNumberOfStandardGaussianWeights - 1; index++) {
    standard_gaussian_weights[index] = \
        (1.0 / std::sqrt(2.0 * M_PI * std::pow(kSigma, 2.0)))
        * std::exp(-std::pow(index, 2.0) / (2.0 * std::pow(kSigma, 2.0)));

    if (index == 0)
      sum_of_weights += standard_gaussian_weights[index];
    else
      sum_of_weights += 2.0 * standard_gaussian_weights[index];
  }

  // Next, normalize these weights to prevent the clipping of the Gaussian
  // curve at the end of the discrete samples from reducing luminance.
  for (int index = 0; index < kNumberOfStandardGaussianWeights - 1; index++) {
    standard_gaussian_weights[index] = \
        standard_gaussian_weights[index] / sum_of_weights;
  }

  // From these weights we calculate the offsets to read interpolated values
  // from.
  const int kNumberOfOptimizedOffsets = \
      std::min(kBlurRadius / 2 + (kBlurRadius % 2), 7);
  float* optimized_gaussian_offsets = reinterpret_cast<float*>(
      std::calloc(kNumberOfOptimizedOffsets, sizeof(float)));

  for (int index = 0; index < kNumberOfOptimizedOffsets; index++) {
    const float kFirstWeight = standard_gaussian_weights[index * 2 + 1];
    const float kSecondWeight = standard_gaussian_weights[index * 2 + 2];
    const float kOptimizedWeight = kFirstWeight + kSecondWeight;

    optimized_gaussian_offsets[index] = \
        (kFirstWeight * (index * 2 + 1) + kSecondWeight * (index * 2 + 2))
        / kOptimizedWeight;
  }

  std::string shader_string;
    shader_string.append(R"(
#include <metal_stdlib>

typedef struct {
  vector_float2 framebufferCoordinate [[attribute(0)]];
  vector_float2 textureCoordinate [[attribute(1)]];
} Vertex;

typedef struct {
  vector_float4 pos [[position]];
  vector_float2 textureCoordinate;
)");

  const int kNumberOfBlurCoordinates = 1 + (kNumberOfOptimizedOffsets * 2);
  for (int i = 0; i < kNumberOfBlurCoordinates; ++i) {
    shader_string.append(
        "\n  vector_float2 blurCoordinates" + std::to_string(i) + ";");
  }

    shader_string.append(R"(
} RasterizerData;

typedef struct  {
  float texelWidthOffset;
  float texelHeightOffset;
} Uniforms;
)");

  shader_string.append(R"(
vertex RasterizerData vertexShader(
    Vertex vert [[stage_in]],
    constant Uniforms& uniforms [[buffer(1)]]) {

  RasterizerData out;
  out.pos = vector_float4(vert.framebufferCoordinate.xy, 0.0, 1.0);
  out.textureCoordinate = vert.textureCoordinate;

  vector_float2 singleStepOffset(uniforms.texelWidthOffset,
                                 uniforms.texelHeightOffset);
)");

  // Inner offset loop.
  shader_string.append(R"(
  out.blurCoordinates0 = out.textureCoordinate.xy;)");
  const char* kInnerOffsetLoopFormat = R"(
  out.blurCoordinates%d = out.textureCoordinate + singleStepOffset * %f;
  out.blurCoordinates%d = out.textureCoordinate - singleStepOffset * %f;)";
  for (int index = 0; index < kNumberOfOptimizedOffsets; index++) {
    const int kFirstIndex = (index * 2) + 1;
    const int kSecondIndex = (index * 2) + 2;
    const float kOptimizedGaussianOffset = optimized_gaussian_offsets[index];

    const int kStringLength = \
        snprintf(NULL, 0, kInnerOffsetLoopFormat, kFirstIndex,
                 kOptimizedGaussianOffset, kSecondIndex,
                 kOptimizedGaussianOffset) + 1;
    char string[kStringLength];
    snprintf(string, kStringLength, kInnerOffsetLoopFormat, kFirstIndex,
             kOptimizedGaussianOffset, kSecondIndex, kOptimizedGaussianOffset);
    shader_string.append(string);
  }

  shader_string.append(R"(
  return out;
}
)");

  // Fragment function.
  shader_string.append(R"(
fragment vector_float4 fragmentShader(
    RasterizerData in [[stage_in]],
    constant Uniforms& uniforms [[buffer(0)]],
    metal::texture2d<float> texture [[texture(0)]]) {
  constexpr metal::sampler sampler (metal::mag_filter::linear,
                                    metal::min_filter::linear);
  vector_float4 sum = vector_float4(0);
)");

  // Inner texture loop.
  const char* kInnerTextureLoopFirstLineFormat = R"(
  sum += texture.sample(sampler, in.blurCoordinates0) * %f;)";
  const int kInnerTextureLoopFirstLineLength = \
      snprintf(NULL, 0, kInnerTextureLoopFirstLineFormat,
      standard_gaussian_weights[0]) + 1;
  char inner_texture_loop_first_line[kInnerTextureLoopFirstLineLength];
  snprintf(inner_texture_loop_first_line, kInnerTextureLoopFirstLineLength,
           kInnerTextureLoopFirstLineFormat, standard_gaussian_weights[0]);
  shader_string.append(inner_texture_loop_first_line);

  const char* kInnerTextureLoopFormat = R"(
  sum += texture.sample(sampler, in.blurCoordinates%d) * %f;
  sum += texture.sample(sampler, in.blurCoordinates%d) * %f;)";
  for (int current_blur_coordinate_index = 0;
       current_blur_coordinate_index < kNumberOfOptimizedOffsets;
       current_blur_coordinate_index++) {
    const float kFirstWeight = \
        standard_gaussian_weights[current_blur_coordinate_index * 2 + 1];
    const float kSecondWeight = \
        standard_gaussian_weights[current_blur_coordinate_index * 2 + 2];
    const float kOptimizedWeight = kFirstWeight + kSecondWeight;

    const int kFirstIndex = current_blur_coordinate_index * 2 + 1;
    const int kSecondIndex = current_blur_coordinate_index * 2 + 2;

    const int kStringLength = snprintf(NULL, 0, kInnerTextureLoopFormat,
                                       kFirstIndex, kOptimizedWeight,
                                       kSecondIndex, kOptimizedWeight) + 1;
    char string[kStringLength];
    snprintf(string, kStringLength, kInnerTextureLoopFormat, kFirstIndex,
             kOptimizedWeight, kSecondIndex, kOptimizedWeight);
    shader_string.append(string);
  }

  // If the number of required samples exceeds the amount we can pass in via
  // varyings, we have to do dependent texture reads in the fragment shader.
  const int kTruekNumberOfOptimizedOffsets = \
      kBlurRadius / 2 + (kBlurRadius % 2);
  if (kTruekNumberOfOptimizedOffsets > kNumberOfOptimizedOffsets) {
    shader_string.append(R"(
  vector_float2 singleStepOffset(uniforms.texelWidthOffset,
                                 uniforms.texelHeightOffset);)");

    const char* kInnerTextureLoopFormat = R"(
  sum += texture.sample(sampler, in.blurCoordinates0 + singleStepOffset * %f) * %f;
  sum += texture.sample(sampler, in.blurCoordinates0 - singleStepOffset * %f) * %f;)";
    for (int current_overlow_texture_read = kNumberOfOptimizedOffsets;
         current_overlow_texture_read < kTruekNumberOfOptimizedOffsets;
         current_overlow_texture_read++) {
      const float kFirstWeight = \
          standard_gaussian_weights[current_overlow_texture_read * 2 + 1];
      const float kSecondWeight = \
          standard_gaussian_weights[current_overlow_texture_read * 2 + 2];
      const float kOptimizedWeight = kFirstWeight + kSecondWeight;
      if (kOptimizedWeight != 0) {
        const float kOptimizedOffset = \
            (kFirstWeight * (current_overlow_texture_read * 2 + 1) +
                kSecondWeight * (current_overlow_texture_read * 2 + 2))
            / kOptimizedWeight;

        const int kStringLength = \
            snprintf(NULL, 0, kInnerTextureLoopFormat, kOptimizedOffset,
                     kOptimizedWeight, kOptimizedOffset, kOptimizedWeight) + 1;
        char string[kStringLength];
        snprintf(string, kStringLength, kInnerTextureLoopFormat,
                 kOptimizedOffset, kOptimizedWeight, kOptimizedOffset,
                 kOptimizedWeight);
        shader_string.append(string);
      }
    }
  }

  shader_string.append(R"(
  return sum;
})");

  std::free(optimized_gaussian_offsets);
  std::free(standard_gaussian_weights);
  return shader_string;
}

@end
