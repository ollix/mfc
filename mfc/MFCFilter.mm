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

#import "mfc/MFCFilter.h"

#include <string>

#import <Metal/Metal.h>
#import <simd/simd.h>

typedef struct {
  float x;
  float y;
} FramebufferCoordinate;

typedef struct {
  float s;
  float t;
} TextureCoordinate;

typedef struct {
  FramebufferCoordinate framebuffer_coordinate;
  TextureCoordinate texture_coordinate;
} Vertex;

const Vertex kVertices[4] = {{{-1.0, -1.0}, {0.0, 1.0}},  // bottom-left,
                             {{-1.0, 1.0}, {0.0, 0.0}},  // top-left
                             {{1.0, -1.0}, {1.0, 1.0}},  // bottom-right,
                             {{1.0, 1.0}, {1.0, 0.0}}};  // top-right

@interface MFCTexture : NSObject {
 @public
  id<MTLTexture> tex;
  id<MTLSamplerState> sampler;
}
@end

@implementation MFCTexture

@end

@implementation MFCFilter

@synthesize device = _device;
@synthesize devicePixelRatio = _devicePixelRatio;
@synthesize tempTexture = _tempTexture;
@synthesize texture = _texture;
@synthesize viewPortSize = _viewPortSize;

- (id)initWithDevice:(id<MTLDevice>)device {
  if (self = [super init]) {
    _device = MTLCreateSystemDefaultDevice();

    // Initializes teh vertex descriptor.
    _vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
    _vertexDescriptor.attributes[0].format = MTLVertexFormatFloat2;
    _vertexDescriptor.attributes[0].bufferIndex = 0;
    _vertexDescriptor.attributes[0].offset = 0;
    _vertexDescriptor.attributes[1].format = MTLVertexFormatFloat2;
    _vertexDescriptor.attributes[1].bufferIndex = 0;
    _vertexDescriptor.attributes[1].offset = sizeof(float) * 2;
    _vertexDescriptor.layouts[0].stride = sizeof(Vertex);
    _vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
  }
  return self;
}

- (void)applyFilter {
}

- (BOOL)compileShaders {
  if (_library != nil) {
    return YES;
  }

  NSError* error = nil;
  NSString* shader = [NSString stringWithUTF8String:[self shader].c_str()];
  _library = [_device newLibraryWithSource:shader
                                   options:NULL
                                     error:&error];

  if (error != nil) {
    return NO;
  }
  return YES;
}

- (void)encodeToCommandQueue:(id<MTLCommandQueue>)commandQueue
              inPlaceTexture:(id<MTLTexture>)texture
                   withWidth:(float)width
                      height:(float)height
            devicePixelRatio:(float)devicePixelRatio {
  _commandQueue = commandQueue;
  _viewPortSize = (vector_uint2){static_cast<uint>(width * devicePixelRatio),
                                 static_cast<uint>(height * devicePixelRatio)};
  _devicePixelRatio = devicePixelRatio;
  _texture = texture;

  // Resets temporary texture if the size is different.
  if (_tempTexture != nil &&
      (_tempTexture.width != _viewPortSize.x ||
       _tempTexture.height != _viewPortSize.y)) {
    _tempTexture = nil;
  }

  // Initializes the temporary texture.
  if (_tempTexture == nil) {
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:_viewPortSize.x
        height:_viewPortSize.y
        mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageShaderRead
                              | MTLTextureUsageRenderTarget
                              | MTLTextureUsageShaderWrite;
    textureDescriptor.storageMode = MTLStorageModePrivate;
    _tempTexture = [_device newTextureWithDescriptor:textureDescriptor];
  }

  if ([self compileShaders]) {
    [self applyFilter];
  }
}

- (void)renderToTexture:(id<MTLTexture>)targetTexture
    withSourceTexture:(id<MTLTexture>)sourceTexture
    vertexFunctionName:(NSString *)vertexFunctionName
    fragmentFunctionName:(NSString *)fragmentFunctionName
    sourceBlendFactor:(MTLBlendFactor)sourceBlendFactor
    destinationBlendFactor:(MTLBlendFactor)destinationBlendFactor
    uniformBuffer:(id<MTLBuffer>)uniformBuffer {
  id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
  [commandBuffer enqueue];

  MTLRenderPassDescriptor *renderPassDescriptor = \
      [MTLRenderPassDescriptor renderPassDescriptor];
  MTLRenderPassColorAttachmentDescriptor* renderPassColorAttachment = \
      renderPassDescriptor.colorAttachments[0];
  renderPassColorAttachment.clearColor = MTLClearColorMake(0.f, 0.f, 0.f, 0.f);
  renderPassColorAttachment.loadAction = MTLLoadActionClear;
  renderPassColorAttachment.storeAction = MTLStoreActionStore;
  renderPassColorAttachment.texture = targetTexture;

  id<MTLRenderCommandEncoder> encoder = [commandBuffer
      renderCommandEncoderWithDescriptor:renderPassDescriptor];
  [encoder setViewport:(MTLViewport)
      {0.0, 0.0, static_cast<double>(_viewPortSize.x),
       static_cast<double>(_viewPortSize.y), 0.0, 1.0}];

  [encoder setVertexBytes:kVertices length:sizeof(kVertices) atIndex:0];
  if (uniformBuffer != nil) {
    [encoder setFragmentBuffer:uniformBuffer offset:0 atIndex:0];
    [encoder setVertexBuffer:uniformBuffer offset:0 atIndex:1];
  }

  // Sets pipeline state.
  MTLRenderPipelineDescriptor* pipelineStateDescriptor =
      [MTLRenderPipelineDescriptor new];
  pipelineStateDescriptor.fragmentFunction = [_library
      newFunctionWithName:fragmentFunctionName];
  pipelineStateDescriptor.vertexFunction = [_library
      newFunctionWithName:vertexFunctionName];
  pipelineStateDescriptor.colorAttachments[0].pixelFormat = \
      MTLPixelFormatRGBA8Unorm;
  pipelineStateDescriptor.vertexDescriptor = _vertexDescriptor;

  MTLRenderPipelineColorAttachmentDescriptor* pipelineColorAttachment = \
      pipelineStateDescriptor.colorAttachments[0];
  pipelineColorAttachment.blendingEnabled = YES;
  pipelineColorAttachment.pixelFormat = MTLPixelFormatRGBA8Unorm;
  pipelineColorAttachment.sourceRGBBlendFactor = sourceBlendFactor;
  pipelineColorAttachment.sourceAlphaBlendFactor = sourceBlendFactor;
  pipelineColorAttachment.destinationRGBBlendFactor = destinationBlendFactor;
  pipelineColorAttachment.destinationAlphaBlendFactor = destinationBlendFactor;

  id<MTLRenderPipelineState> pipelineState = [_device
      newRenderPipelineStateWithDescriptor:pipelineStateDescriptor
      error:nil];

  [encoder setRenderPipelineState:pipelineState];

  [encoder setFragmentTexture:sourceTexture atIndex:0];
  [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
              vertexStart:0
              vertexCount:4];

  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
}

- (std::string)shader {
  return "";
}

@end
