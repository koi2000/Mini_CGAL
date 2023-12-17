#include "DecompressTool.cuh"
#include <fstream>
#include <unistd.h>

// Write a given number of bits in a buffer.
void writeBits(uint32_t data, unsigned i_nbBits, char* p_dest, unsigned& i_bitOffset, size_t& offset) {
    assert(i_nbBits <= 25);

    uint32_t dataToAdd = data << (32 - i_nbBits - i_bitOffset);
    // Swap the integer bytes because the x86 architecture is little endian.
    dataToAdd = __builtin_bswap32(dataToAdd);  // Call a GCC builtin function.

    // Write the data.
    *(uint32_t*)p_dest |= dataToAdd;

    // Update the size and offset.
    offset += (i_bitOffset + i_nbBits) / 8;
    i_bitOffset = (i_bitOffset + i_nbBits) % 8;
}

/**
 * Read a given number of bits in a buffer.
 */
uint32_t readBits(unsigned i_nbBits, char* p_src, unsigned& i_bitOffset, size_t& offset) {
    assert(i_nbBits <= 25);

    // Build the mask.
    uint32_t mask = 0;
    for (unsigned i = 0; i < 32 - i_bitOffset; ++i)
        mask |= 1 << i;
    // Swap the mask bytes because the x86 architecture is little endian.
    mask = __builtin_bswap32(mask);  // Call a GCC builtin function.

    uint32_t data = *(uint32_t*)p_src & mask;

    // Swap the integer bytes because the x86 architecture is little endian.
    data = __builtin_bswap32(data);  // Call a GCC builtin function.

    data >>= 32 - i_nbBits - i_bitOffset;

    // Update the size and offset.
    offset += (i_bitOffset + i_nbBits) / 8;
    i_bitOffset = (i_bitOffset + i_nbBits) % 8;

    return data;
}

// Write a floating point number in the data buffer.
void DeCompressTool::writeFloat(float f, int* offset) {
    *(float*)(buffer + *offset) = f;
    *offset += sizeof(float);
}

/**
 * Read a floating point number in the data buffer.
 */
float DeCompressTool::readFloat(int* offset) {
    float f = *(float*)(buffer + *offset);
    *offset += sizeof(float);
    return f;
}

// Write a floating point number in the data buffer.
void DeCompressTool::writePoint(MCGAL::Point& p, int* offset) {
    for (unsigned i = 0; i < 3; ++i) {
        writeFloat(p[i],offset);
    }
}

// Write a floating point number in the data buffer.
MCGAL::Point DeCompressTool::readPoint(int* offset) {
    float coord[3];
    for (unsigned i = 0; i < 3; ++i) {
        coord[i] = readFloat(offset);
    }
    MCGAL::Point pt(coord[0], coord[1], coord[2]);
    return pt;
}

/**
 * Read an integer in the data buffer.
 */
int DeCompressTool::readInt(int* offset) {
    int i = *(int*)(buffer + *offset);
    *offset += sizeof(int);
    return i;
}

// Write an integer in the data buffer
void DeCompressTool::writeInt(int i, int* offset) {
    *(int*)(buffer + *offset) = i;
    *offset += sizeof(int);
}

/**
 * Read a 16 bits integer in the data buffer.
 */
int16_t DeCompressTool::readInt16(int* offset) {
    int16_t i = *(int16_t*)(buffer + *offset);
    *offset += sizeof(int16_t);
    return i;
}

// Write a 16 bits integer in the data buffer
void DeCompressTool::writeInt16(int16_t i,int* offset) {
    *(int16_t*)(buffer + *offset) = i;
    *offset += sizeof(int16_t);
}

/**
 * Read a 16 bits integer in the data buffer.
 */
uint16_t DeCompressTool::readuInt16(int* offset) {
    uint16_t i = *(uint16_t*)(buffer + *offset);
    *offset += sizeof(uint16_t);
    return i;
}

// Write a 16 bits integer in the data buffer
void DeCompressTool::writeuInt16(uint16_t i,int* offset) {
    *(uint16_t*)(buffer + *offset) = i;
    *offset += sizeof(uint16_t);
}

/**
 * Read a byte in the data buffer.
 */
unsigned char DeCompressTool::readChar(int* offset) {
    unsigned char i = *(unsigned char*)(buffer + *offset);
    *offset += sizeof(unsigned char);
    return i;
}

// Write a byte in the data buffer
void DeCompressTool::writeChar(unsigned char i,int* offset) {
    *(unsigned char*)(buffer + *offset) = i;
    *offset += sizeof(unsigned char);
}