// Copyright (c) 2026 Ollix. All rights reserved.
//
// Standalone macOS regression test (run from the repository root):
// xcrun clang++ -std=c++11 -fobjc-arc -I. test/gaussian_blur_shader_test.mm \
//   mfc/MFCFilter.mm mfc/MFCGaussianBlurFilter.mm -framework Foundation \
//   -framework Metal -o /tmp/mfc-shader-test
// /tmp/mfc-shader-test
// Use --dump to compare the generated shader text before and after a change.

#import "mfc/MFCGaussianBlurFilter.h"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::cerr << "A Metal device is required.\n";
      return 1;
    }
    const bool dump = argc == 2 && std::string(argv[1]) == "--dump";
    int cases = 0;
    for (float radius : {0.f, 1.f, 2.f, 7.f, 14.f, 15.f, 16.f, 31.f}) {
      for (float ratio : {1.f, 2.f, 3.f}) {
        for (float sigma : {1.f, 3.f}) {
          MFCGaussianBlurFilter *filter = [[MFCGaussianBlurFilter alloc]
              initWithDevice:device blurRadius:radius andSigma:sigma];
          filter.devicePixelRatio = ratio;
          const std::string shader = [filter shader];
          if (radius == 0) {
            if (!shader.empty()) return 1;
          } else {
            NSError *error = nil;
            id<MTLLibrary> library = [device
                newLibraryWithSource:[NSString stringWithUTF8String:shader.c_str()]
                options:nil error:&error];
            if (library == nil || error != nil ||
                [library newFunctionWithName:@"vertexShader"] == nil ||
                [library newFunctionWithName:@"fragmentShader"] == nil) {
              std::cerr << "Shader failed: radius=" << radius
                        << ", ratio=" << ratio << ", sigma=" << sigma << '\n';
              if (error != nil) std::cerr << error.description.UTF8String << '\n';
              return 1;
            }
          }
          if (dump) std::cout << radius << ',' << ratio << ',' << sigma
                              << '\n' << shader << '\n';
          ++cases;
        }
      }
    }
    std::cerr << cases << " shader cases passed.\n";
  }
  return 0;
}
