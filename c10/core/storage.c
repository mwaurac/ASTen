#include "storage.h"

#include <stdio.h>

#ifdef __CUDA__
#include <cuda_runtime.h>
#endif

Storage* storage_new(size_t numel, DType dtype, Device device) {
    Storage* storage = (Storage*)malloc(sizeof(Storage));
    if (!storage) return NULL;
    
    storage->numel=numel;
    storage->dtype=dtype;
    storage->device=device;
    storage->refcount=1;
    
    size_t size = numel * dtype_size(dtype);
    
    if (device.type == DEVICE_CPU) {
        storage->data = malloc(size);
        if (!storage->data) {
            free(storage);
            return NULL;
        }
        memset(storage->data, 0, size);
    }
    
    #ifdef __CUDA__
    else if (device.type == DEVICE_CUDA) {
        cudaError_t err = cudaMalloc(&storage->data, size);
        if (error != cudaSuccess) {
            free(storage);
            return NULL;
        }
        cudaMemset(storage->data, 0, size);
    }
    #endif
    else {
        free(storage);
        return NULL;
    }
    return storage;
}

void storage_retain(Storage *storage) {
    if (storage) {
        storage->refcount++;
    }
}

void storage_release(Storage *storage) {
    if (!storage) return;
    
    storage->refcount--;
    if (storage->refcount <=0) {
        if (storage->data) {
            if (storage->device.type == DEVICE_CPU) {
                free(storage->data);
            }
            #ifdef __CUDA__
            else if (storage->device.type == DEVICE_CUDA) {
                cudaFree(storage->data);
            }
            #endif
        }
        free(storage);
    }
}

void* storage_data(Storage *storage) {
    return storage ? storage->data : NULL;
}

size_t storage_element_size(Storage *storage) {
    return storage ? dtype_size(storage->dtype) : 0;
}

size_t storage_size_bytes(Storage *storage) {
    return storage ? (storage->numel * dtype_size(storage->dtype)) : 0;
}