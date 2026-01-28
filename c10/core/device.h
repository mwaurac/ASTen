#ifndef C10_DEVICE_H
#define C10_DEVICE_H

typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
    DEVICE_UNKOWNN
} DeviceType;

typedef struct {
   DeviceType type;
   int index ;
} Device;

static inline Device device_cpu () {
    Device dev = {DEVICE_CPU, 0};
    return dev;
}

static inline Device device_cuda (int index) {
    Device dev = {DEVICE_CUDA, index};
    return dev;
}

static inline const char* device_name(Device device) {
    switch (device.type) {
        case DEVICE_CPU: return "cpu";
        case DEVICE_CUDA: return "cuda";
        default: return "unknown";
    }
}

static inline int device_equal(Device a, Device b) {
    return a.type == b.type && a.index == b.index;
}

#endif