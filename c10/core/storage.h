#ifndef C10_STORAGE_H
#define C10_STORAGE_H

#include "dtype.h"
#include "device.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    void* data;
    size_t numel;
    DType dtype;
    Device device;
    int refcount;
} Storage;

Storage* storage_new(size_t numel, DType dtype, Device device);
void storage_retain(Storage* storage);
void storage_release(Storage *storage);
void* storage_data(Storage* storage);
size_t storage_element_size(Storage* storage);
size_t storage_size_bytes(Storage* storage);

#endif