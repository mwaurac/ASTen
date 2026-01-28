#ifndef C10_DTYPE_H
#define C10_DTYPE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_INT64
    // TODO: ADD OTHERS LIKE BFLOAT16
} DType;

static inline size_t dtype_size (DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32: return sizeof(int32_t);
        case DTYPE_INT64: return sizeof(int64_t);
        default: return 0;
    }
}

static inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return "float32";
        case DTYPE_FLOAT64: return "float64";
        case DTYPE_INT32: return "int32";
        case DTYPE_INT64: return "int64";
        default: return "unknown";
    }
}


#endif 