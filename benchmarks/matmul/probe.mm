#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

constexpr int rows = 128, inner = 768, cols = 50304;
MPSMatrix *matrix(id<MTLBuffer> buffer, int r, int c) {
    auto* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:r columns:c rowBytes:c*sizeof(float) dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];
}
float nextFloat(uint32_t& seed) {
    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
    return float(seed >> 8) * (1.0f/8388608.0f) - 1.0f;
}
int main(int argc, char** argv) { @autoreleasepool {
    if(argc!=2) return 1;
    auto gpu=MTLCreateSystemDefaultDevice();
    auto queue=[gpu newCommandQueue];
    NSError* error=nil;
    auto source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]] encoding:NSUTF8StringEncoding error:&error];
    auto options=[[MTLCompileOptions alloc] init];
    options.mathMode=MTLMathModeSafe;
    auto library=[gpu newLibraryWithSource:source options:options error:&error];
    auto fn=[library newFunctionWithName:@"_Z22matmul_forward_kernel4PfPKfS1_S1_ii"];
    auto pipeline=[gpu newComputePipelineStateWithFunction:fn error:&error];
    if(!pipeline){fprintf(stderr,"%s\n", error.localizedDescription.UTF8String);return 2;}
    auto alloc=[&](size_t count){return [gpu newBufferWithLength:count*sizeof(float) options:MTLResourceStorageModeShared];};
    auto a=alloc(rows*inner), w=alloc(cols*inner), outTranslated=alloc(rows*cols), outApple=alloc(rows*cols);
    auto ap=(float*)a.contents, wp=(float*)w.contents;
    uint32_t seed=20261108;
    for(int i=0;i<rows*inner;++i) ap[i]=nextFloat(seed);
    for(int i=0;i<cols*inner;++i) wp[i]=nextFloat(seed);
    std::fill_n((float*)outTranslated.contents,rows*cols,NAN);
    std::fill_n((float*)outApple.contents,rows*cols,NAN);
    auto mps=[[MPSMatrixMultiplication alloc] initWithDevice:gpu transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:cols interiorColumns:inner alpha:1 beta:0];
    auto am=matrix(a,rows,inner), wm=matrix(w,cols,inner), cm=matrix(outApple,rows,cols);
    printf("device,%s\n",gpu.name.UTF8String);
    for(int pair=-3;pair<12;++pair) {
        for(int leg=0;leg<2;++leg){@autoreleasepool{
            bool native=((pair+3)%2)==leg;
            auto t0=std::chrono::steady_clock::now();
            auto cb=[queue commandBuffer];
            if(native){
                [mps encodeToCommandBuffer:cb leftMatrix:am rightMatrix:wm resultMatrix:cm];
            } else {
                auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:pipeline];
                [enc setBuffer:outTranslated offset:0 atIndex:0];[enc setBuffer:a offset:0 atIndex:1];[enc setBuffer:w offset:0 atIndex:2];[enc setBuffer:nil offset:0 atIndex:3];
                [enc setBytes:&inner length:4 atIndex:4];[enc setBytes:&cols length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(rows/128,cols/128,1) threadsPerThreadgroup:MTLSizeMake(16,16,1)];[enc endEncoding];
            }
            [cb commit];[cb waitUntilCompleted];
            if(cb.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"%s\n",cb.error.description.UTF8String);return 3;}
            double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            printf("row,%d,%d,%s,%.7f,%.7f\n", pair,leg,native?"mps":"translated",(cb.GPUEndTime-cb.GPUStartTime)*1000.,wall);
        }}
    }
    float *tp=(float*)outTranslated.contents,*mp=(float*)outApple.contents;
    double maxErr=0., sumsq=0., maxCpuTranslated=0., maxCpuApple=0.;
    int failures=0;
    for(int i=0;i<rows*cols;++i){
        double err=fabs(double(tp[i])-mp[i]);
        if(!isfinite(tp[i])||!isfinite(mp[i])||err>0.0003+0.00002*fabs(tp[i])) ++failures;
        maxErr=std::max(maxErr,err); sumsq+=err*err;
    }
    for(int i=0;i<128;++i){
        const int r=(i*67)%rows,c=(i*31721)%cols;
        double ref=0.;for(int k=0;k<inner;++k)ref+=double(ap[r*inner+k])*wp[c*inner+k];
        maxCpuTranslated=std::max(maxCpuTranslated,fabs(tp[r*cols+c]-ref));
        maxCpuApple=std::max(maxCpuApple,fabs(mp[r*cols+c]-ref));
    }
    printf("result,%d,%d,%.9g,%.9g,%.9g,%.9g\n",rows*cols,failures,maxErr,sqrt(sumsq/(rows*cols)),maxCpuTranslated,maxCpuApple);
    return failures || !(maxCpuTranslated<0.0005 && maxCpuApple<0.0005) ? 4:0;
}}
