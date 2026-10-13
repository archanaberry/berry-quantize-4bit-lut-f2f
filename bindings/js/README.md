# Node.js binding

This binding uses Koffi to call the opaque Berryfour C ABI directly; it does not require a custom Node-API addon.

```sh
make
cd bindings/js
npm install
npm test
```

Set `BERRYBITS_LIBRARY` to select a non-default shared-library path.
