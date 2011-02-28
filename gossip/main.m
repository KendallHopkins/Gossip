//
//  main.m
//  gossip
//
//  Created by Kendall Hopkins on 2/19/11.
//  Copyright 2011 SoftwareElves. All rights reserved.
//

#import <Foundation/Foundation.h>

#import <RROpenCL/RROpenCL.h>

#include "main.h"

#include <stdint.h>

void dumpDawg( const AIDictionaryStorageNode * dictionaryForest, uint32_t currentOffset, uint32_t currentLevel, char * word )
{
    const AIDictionaryStorageNode * dictionaryNode = &dictionaryForest[currentOffset];
    if( dictionaryNode->is_word ) {
        word[currentLevel] = 0;
        printf("%s\n", word);
    }
    for( int i = 0; i < 26; i++ ) {
        uint16_t offset = dictionaryNode->children_offset[i];
        if( offset ) {
            word[currentLevel] = I_TO_C(i);
            dumpDawg( dictionaryForest, offset, currentLevel + 1, word );
        }
    }
}

uint32_t countHints( const AIDictionaryStorageNode * dictionaryForest, uint32_t currentOffset, uint32_t currentLevel )
{
    const AIDictionaryStorageNode * dictionaryNode = &dictionaryForest[currentOffset];
    if( currentLevel < AI_HINT_SIZE ) {
        uint32_t count = 0;
        for( int i = 0; i < 26; i++ ) {
            uint16_t offset = dictionaryNode->children_offset[i];
            if( offset ) {
                count += countHints( dictionaryForest, offset, currentLevel + 1 );
            }
        }
        return count;
    } else {
        return 1;
    }
}

void writeHints( const AIDictionaryStorageNode * dictionaryForest, uint32_t currentOffset, uint32_t currentLevel, AIHint * hintArray, uint32_t * currentHintOffset, char * hint )
{
    const AIDictionaryStorageNode * dictionaryNode = &dictionaryForest[currentOffset];
    if( currentLevel < AI_HINT_SIZE ) {
        for( int i = 0; i < 26; i++ ) {
            uint16_t offset = dictionaryNode->children_offset[i];
            if( offset ) {
                hint[currentLevel] = I_TO_C(i);
                writeHints(dictionaryForest, offset, currentLevel + 1, hintArray, currentHintOffset, hint);
            }
        }
    } else {
        hintArray[*currentHintOffset].node_offset = currentOffset;
        for( int i = 0; i < AI_HINT_SIZE; i++ ) {
            hintArray[*currentHintOffset].prefix[i] = hint[i];            
        }
        (*currentHintOffset)++;
    }
}

NSData * calculateHints( NSData * dictionary )
{
    /* find all hints */
    const AIDictionaryStorageNode * dictionaryNodes = [dictionary bytes];
    
    uint32_t hintCount = countHints( dictionaryNodes, 0, 0 );

    NSMutableData * hintData = [NSMutableData dataWithLength:hintCount*sizeof( AIHint )];
    AIHint * hintArray = [hintData mutableBytes];
    uint32_t currentHintOffset = 0;
    char hint[AI_HINT_SIZE];
    
    writeHints(dictionaryNodes, 0, 0, hintArray, &currentHintOffset, hint);

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
    [program buildWithIncludePath:@"/Users/ken/Library/Developer/Xcode/DerivedData/gossip-fkwoheldcvsttrfrmalckylclazk/Build/Products/Debug"];
    RRCLKernel * mainKernel = [[RRCLKernel alloc] initWithKernelName:@"search" inProgram:program];
    
    NSData * hintData = calculateHints( dictionary );
    size_t hintSize = [hintData length] / sizeof( AIHint ); /* ~300 */
    NSData * hintSizeData = [NSData dataWithBytesNoCopy:&hintSize length:sizeof( hintSize ) freeWhenDone:NO];
    
    /* send rest to GPU */
    RRCLBuffer * dictionaryBuffer = [[RRCLBuffer alloc] initReadOnlyWithContext:context size:[dictionary length]];
    RRCLBuffer * hintBuffer = [[RRCLBuffer alloc] initReadOnlyWithContext:context size:[hintData length]];
    size_t wordOutputSize = 1024*1024*8;
    RRCLBuffer * wordOutputBuffer = [[RRCLBuffer alloc] initWriteOnlyWithContext:context size:wordOutputSize];
    NSData * wordOutputBufferSize = [NSData dataWithBytesNoCopy:&wordOutputSize length:sizeof(wordOutputSize) freeWhenDone:NO];
    
    //dictionary args
    [mainKernel setArg:0 toBuffer:dictionaryBuffer];
    
    //hint args
    [mainKernel setArg:1 toBuffer:hintBuffer];
    [mainKernel setArg:2 toData:hintSizeData];
    
    //output buffer args
    RRCLBuffer * wordOutputOffsetBuffer = [[RRCLBuffer alloc] initReadWriteWithContext:context size:sizeof(cl_int)];
    [mainKernel setArg:3 toBuffer:wordOutputBuffer];
    [mainKernel setArg:4 toData:wordOutputBufferSize];
    [mainKernel setArg:5 toBuffer:wordOutputOffsetBuffer];
    
    NSLog(@"first-start");
    [commandQueue enqueueWriteBuffer:dictionaryBuffer blocking:NO offset:0 data:dictionary];
    [commandQueue enqueueWriteBuffer:hintBuffer blocking:NO offset:0 data:hintData];
    cl_int wordOutputOffset = 0;;
    [commandQueue enqueueWriteBuffer:wordOutputOffsetBuffer blocking:NO offset:0 data:[NSData dataWithBytes:&wordOutputOffset length:sizeof(wordOutputOffset)]];
    [commandQueue enqueueNDRangeKernel:mainKernel globalWorkSize:hintSize];
    [commandQueue enqueueReadBuffer:wordOutputBuffer blocking:NO];
    [commandQueue finish];
    NSLog(@"first-end");
    
    NSLog(@"second-start");
    //reset wordOutputOffset
    wordOutputOffset = 0;
    [commandQueue enqueueWriteBuffer:wordOutputOffsetBuffer blocking:NO offset:0 data:[NSData dataWithBytes:&wordOutputOffset length:sizeof(wordOutputOffset)]];
    
    //enqueue kernel
    [commandQueue enqueueNDRangeKernel:mainKernel globalWorkSize:hintSize];
    
    //get wordOutputOffset
    wordOutputOffset = *(const cl_int *)[[commandQueue enqueueReadBuffer:wordOutputOffsetBuffer blocking:YES] bytes];
    
    //read wordOutput
    NSData * wordOutputData = [commandQueue enqueueReadBuffer:wordOutputBuffer blocking:YES offset:0 length:wordOutputOffset];

    NSLog(@"second-end");
   
    char output[wordOutputOffset+1];
    output[wordOutputOffset]=0;
    memcpy(output, [wordOutputData bytes], wordOutputOffset);
    NSLog(@"%s", output );
    
    [pool drain];
    return 0;
}

