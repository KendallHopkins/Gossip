//
//  main.m
//  gossip
//
//  Created by Kendall Hopkins on 2/19/11.
//  Copyright 2011 SoftwareElves. All rights reserved.
//

#warning Make them into locals

#import <Foundation/Foundation.h>

#import <RROpenCL/RROpenCL.h>

#include "main.h"

#include <stdint.h>

NSData * calculateHints( NSData * dictionary )
{
    /* find all hints */
    const AIDictionaryStorageNode * dictionaryNodes = [dictionary bytes];
    uint32_t hintCount = 0;
    for( int i = 0; i < 26; i++ ) {
        for( int j = 0; j < 26; j++ ) {
            uint16_t first_offset = dictionaryNodes[0].children_offset[i];
            if( first_offset ) {
                uint16_t second_offset = dictionaryNodes[first_offset].children_offset[j];
                if( second_offset ) {
                    hintCount++;
                }
            }
        }        
    }
    NSMutableData * hintData = [NSMutableData dataWithLength:hintCount*sizeof( AIHint )];
    AIHint * hintArray = [hintData mutableBytes];
    hintCount = 0;
    for( int i = 0; i < 26; i++ ) {
        for( int j = 0; j < 26; j++ ) {
            uint16_t first_offset = dictionaryNodes[0].children_offset[i];
            if( first_offset ) {
                uint16_t second_offset = dictionaryNodes[first_offset].children_offset[j];
                if( second_offset ) {
                    hintArray[hintCount].node_offset = second_offset;
                    hintArray[hintCount].prefix[0] = I_TO_C( i );
                    hintArray[hintCount].prefix[1] = I_TO_C( j );
                    hintCount++;
                }
            }
        }        
    }
    return hintData;
}

int main (int argc, const char * argv[])
{
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    
    NSError * error = nil;
    NSString * programCode = [NSString stringWithContentsOfFile:@"main.cl" encoding:NSASCIIStringEncoding error:&error];
    
    NSData * dictionary = [NSData dataWithContentsOfFile:@"dictionary.dawg"];
    RRCLDevice * mainDevice = [RRCLDevice defaultDeviceOfType:CL_DEVICE_TYPE_CPU];
    NSLog(@"Using %@", mainDevice );
    RRCLContext * context = [[RRCLContext alloc] initWithDevices:[NSArray arrayWithObject:mainDevice]];
	RRCLCommandQueue * commandQueue = [[RRCLCommandQueue alloc] initWithContext:context device:mainDevice];
    RRCLProgram * program = [[RRCLProgram alloc] initWithSource:programCode inContext:context];
    [program build];
    RRCLKernel * mainKernel = [[RRCLKernel alloc] initWithKernelName:@"main" inProgram:program];
    
    NSData * hintData = calculateHints( dictionary );
    size_t hintSize = [hintData length] / sizeof( AIHint ); /* ~300 */
    NSData * hintSizeData = [NSData dataWithBytesNoCopy:&hintSize length:sizeof( hintSize ) freeWhenDone:NO];
    
    /* send rest to GPU */
    RRCLBuffer * dictionaryBuffer = [[RRCLBuffer alloc] initReadOnlyWithContext:context size:[dictionary length]];
    RRCLBuffer * hintBuffer = [[RRCLBuffer alloc] initReadOnlyWithContext:context size:[hintData length]];
    size_t wordOutputSize = 100000;
    RRCLBuffer * wordOutputBuffer = [[RRCLBuffer alloc] initWriteOnlyWithContext:context size:wordOutputSize];
    NSData * wordOutputBufferSize = [NSData dataWithBytesNoCopy:&wordOutputSize length:sizeof(wordOutputSize) freeWhenDone:NO];
    
    //dictionary args
    [mainKernel setArg:0 toBuffer:dictionaryBuffer];
    
    //hint args
    [mainKernel setArg:1 toBuffer:hintBuffer];
    [mainKernel setArg:2 toData:hintSizeData];
    [mainKernel setArg:3 toShareWithSize:[NSNumber numberWithUnsignedInt:sizeof( cl_uint )]];
    
    //output buffer args
    [mainKernel setArg:4 toBuffer:wordOutputBuffer];
    [mainKernel setArg:5 toData:wordOutputBufferSize];
    [mainKernel setArg:6 toShareWithSize:[NSNumber numberWithUnsignedInt:sizeof( cl_uint )]];
    size_t localWorkSize = 1;
    
    [commandQueue enqueueWriteBuffer:dictionaryBuffer blocking:NO offset:0 data:dictionary];
    [commandQueue enqueueWriteBuffer:hintBuffer blocking:NO offset:0 data:hintData];
    [commandQueue enqueueNDRangeKernel:mainKernel globalWorkSize:localWorkSize localWorkSize:localWorkSize];
    NSData * wordOutputData = [commandQueue enqueueReadBuffer:wordOutputBuffer blocking:NO];
    [commandQueue finish];
   
    char output[wordOutputSize];
    output[wordOutputSize-1]=0;
    memcpy(output, [wordOutputData bytes], wordOutputSize-1);
    NSLog(@"%s", output );
    
    [pool drain];
    return 0;
}

