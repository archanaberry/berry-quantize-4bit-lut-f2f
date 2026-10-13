# C3 binding

`berrybits.c3` declares the opaque native C ABI using C3 `extern fn` declarations. `example.c3` demonstrates training a tiny student, running GEMV, and saving the model.

Build the native library first with `make`. Then use the C3 compiler's shared-library link option for `build/libberrybits.so` and include this directory on the module path. C3 is not installed in the development VM, so this source binding cannot be compiler-verified here.
