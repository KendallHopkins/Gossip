
#define LETTER_COUNT 26
#define C_TO_I( c ) ( c - 'a' )
#define I_TO_C( i ) ( i + 'a' )


typedef struct AIDictionaryStorageNode AIDictionaryStorageNode;
struct AIDictionaryStorageNode {
    cl_uchar is_word;
    cl_uchar min_depth;
    cl_uchar max_depth;
    cl_uchar _null1;
    cl_uint bitmap;
    cl_ushort children_offset[LETTER_COUNT];
    cl_uint _null2;
};

#define AI_HINT_SIZE 2
typedef struct AIHint AIHint;
struct AIHint {
    cl_uchar prefix[AI_HINT_SIZE];
    cl_ushort node_offset;
};