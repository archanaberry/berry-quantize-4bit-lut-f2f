#include "berrybits_opencl.h"

#include <stdio.h>

int main(void) {
    int available = berrybits_opencl_available();
    printf("OpenCL GPU runtime: %s\n", available ? "available" : "unavailable");
    return 0;
}
