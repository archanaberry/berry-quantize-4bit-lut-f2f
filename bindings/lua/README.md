# LuaJIT FFI binding

Build the C library first, then run from the repository root:

```sh
LUA_PATH="bindings/lua/?.lua;;" luajit bindings/lua/test.lua
```

Set `BERRYBITS_LIBRARY` to select another shared-library path. This binding uses LuaJIT's built-in FFI; standard Lua without FFI needs a compiled C module.
