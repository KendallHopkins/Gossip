#include "apiToCL.h" /* allow sharing of .h file */
#include "main.h"

//#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

void writeToWordOutputBuffer( __global char * wordOutput, __global uint * wordOutputOffset, const char * word, uint wordSize );

void writeToWordOutputBuffer( __global char * wordOutput, __global uint * wordOutputOffset, const char * word, uint wordSize )
{
    /* create a space in the output buffer, by atomically pushing back the offset */
    uint newWordOffset = atom_add( wordOutputOffset, wordSize+1 );

    for( uint j = 0; j < wordSize; j++ ) {
        wordOutput[newWordOffset+j] = word[j];
    }
    wordOutput[newWordOffset+wordSize] = '\n';
}

/* There be dragons here */
__kernel void search( __global AIDictionaryStorageNode * node_array,
                      __global AIHint * hint_array,
                      size_t hint_size,
                      __global char * wordOutput,
                      size_t wordOutputSize,
                      __global uint * wordOutputOffset )
{
    size_t global_id = get_global_id(0);

    AIHint cellHint = hint_array[global_id];

    size_t pattern_size = 16;
    char pattern[pattern_size+1];
    pattern[0] = '*';
    pattern[1] = 'a';
    pattern[2] = 'b';
    pattern[3] = 'c';
    pattern[4] = '*';
    pattern[5] = '*';
    pattern[6] = '*';
    pattern[7] = '*';
    pattern[8] = '*';
    pattern[9] = '*';
    pattern[10] = '*';
    pattern[11] = '*';
    pattern[12] = '*';
    pattern[13] = '*';
    pattern[14] = '*';
    pattern[15] = '*';
    pattern[16] = 0;
    char wordBuffer[pattern_size];

    //scope stuff
    AIDictionaryStorageNode scopeNode[pattern_size+1];
    uchar scopeI[pattern_size+1];

    //process hint
    for( int i = 0; i < AI_HINT_SIZE; i++ ) {
        if( pattern[i] != '*' && pattern[i] != cellHint.prefix[i] ) {
            return; //hint isn't going to yeild anything w/ the pattern
        }
        wordBuffer[i] = cellHint.prefix[i];
    }
    
    //bootstrap first scope
    int currentScope = 0;
    scopeI[0] = 0;
    scopeNode[0] = node_array[cellHint.node_offset];
    
    //check if the hint is a word itself
    if( scopeNode[0].is_word ) {
        writeToWordOutputBuffer( wordOutput, wordOutputOffset, wordBuffer, AI_HINT_SIZE );
    }
    
    while( currentScope >= 0 ) {
        ushort nextNodeOffset = 0;

        char currentPattern = pattern[AI_HINT_SIZE+currentScope];
        if( currentPattern >= 'a' && currentPattern <= 'z' ) {
            if( ! scopeI[currentScope] ) {
                wordBuffer[AI_HINT_SIZE+currentScope] = currentPattern;
                nextNodeOffset = scopeNode[currentScope].children_offset[C_TO_I(currentPattern)];
                scopeI[currentScope] = 1;
            } else {
                nextNodeOffset = 0;
            }
        } else if( currentPattern == '*' ) {
            for( nextNodeOffset = 0; scopeI[currentScope] < LETTER_COUNT && ! nextNodeOffset; scopeI[currentScope]++ ) {
                nextNodeOffset = scopeNode[currentScope].children_offset[scopeI[currentScope]];
                if( nextNodeOffset ) {
                    wordBuffer[AI_HINT_SIZE+currentScope] = I_TO_C( scopeI[currentScope] );
                }
            }
        }

        if( nextNodeOffset ) {
            //start new scope
            currentScope++;
            scopeNode[currentScope] = node_array[nextNodeOffset];
            scopeI[currentScope] = 0;
            
            //write out word if
            if( scopeNode[currentScope].is_word )
                writeToWordOutputBuffer( wordOutput, wordOutputOffset, wordBuffer, AI_HINT_SIZE+currentScope );
        } else {
            //back out of scope
            currentScope--;
        }
    }    
}
