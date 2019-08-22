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

#include <string>

#import <Metal/Metal.h>
#import <simd/simd.h>

@interface MFCFilter : NSObject {
 @private
  id<MTLCommandBuffer> _commandBuffer;
  id<MTLCommandQueue> _commandQueue;
  id<MTLFunction> _defaultVertexFunction;
  id<MTLDevice> _device;
  float _devicePixelRatio;
  id<MTLLibrary> _library;
  id<MTLTexture> _tempTexture;
  id<MTLTexture> _texture;
  vector_uint2 _viewPortSize;
}

@property (retain) id<MTLDevice> device;
@property (assign) float devicePixelRatio;
@property (retain) id<MTLTexture> tempTexture;
@property (retain) id<MTLTexture> texture;
@property (assign) vector_uint2 viewPortSize;

- (id)initWithDevice:(id<MTLDevice>)device;

- (void)applyFilter;

- (BOOL)compileShaders;

- (void)encodeToCommandQueue:(id<MTLCommandQueue>)commandQueue
              inPlaceTexture:(id<MTLTexture>)texture
                   withWidth:(float)width
                      height:(float)height
            devicePixelRatio:(float)devicePixelRatio;

- (void)renderToTexture:(id<MTLTexture>)targetTexture
    withSourceTexture:(id<MTLTexture>)sourceTexture
    vertexFunctionName:(NSString *)vertexFunctionName
    fragmentFunctionName:(NSString *)fragmentFunctionName
    sourceBlendFactor:(MTLBlendFactor)sourceBlendFactor
    destinationBlendFactor:(MTLBlendFactor)destinationBlendFactor
    uniformBuffer:(id<MTLBuffer>)uniformBuffer;

- (std::string)shader;

@end
