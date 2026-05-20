# Contributing to Shivya

Thank you for contributing to the Shivya distributed computing substrate! 

## Code Discipline
- **Zero External Dependencies**: All core crate layers (0 through 4) MUST remain pure, zero-dependency Rust. External dependencies are restricted to telemetry bindings (such as `wasm-bindgen` and `getrandom` under `crates/telemetry_wasm`).
- **Topological & Mathematical Invariants**: All structural and variational operations must mathematically respect discrete exterior calculus foundations (e.g. $d \circ d = 0$) and CFL numerical stability bounds.
- **Pre-allocated Array Bounding**: Dynamic heap allocation/resizing routines should be avoided on active paths; prefer using pre-allocated index pools and bounded vectors.

## Developer Workflow
To check your changes, run:
```bash
cargo check --workspace
cargo test --workspace
```

## Pull Request Guidelines
- Verify all unit and integration tests compile and pass.
- Maintain microsecond performance bounds.
- Document any architectural alterations in the corresponding documentation folders.
