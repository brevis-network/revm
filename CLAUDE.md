# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REVM is a highly efficient Rust implementation of the Ethereum Virtual Machine (EVM). It serves both as:
1. A standard EVM for executing Ethereum transactions
2. A framework for building custom EVM variants (like Optimism's op-revm)

The project is used by major Ethereum infrastructure including Reth, Foundry, Hardhat, Optimism, Scroll, and many zkVMs.

## Build and Development Commands

### Essential Commands

```bash
# Build the project
cargo build
cargo build --release

# Run all tests. `--features serde` is not optional: `tests/shared_memory_serde.rs` is a
# `required-features = ["serde"]` target, so without it cargo skips the target entirely.
cargo nextest run --workspace --features serde
cargo nextest run --workspace                       # the default feature set
cargo nextest run --workspace --no-default-features # the no_std-shaped surface

# `memory_limit` gates the two MLOAD/MSTORE `MemoryLimitOOG` regression tests; run this
# before touching memory expansion.
cargo nextest run -p revm-interpreter --features serde,memory_limit

# Lint and format. `ci.yml` says `--all-features`, which does not build here: it selects
# `revm-precompile`'s `bn` backend, which does not compile against the pinned `substrate-bn`
# (`cannot add AffineG1 to AffineG1`). Use `--features serde` locally.
cargo clippy --workspace --all-targets --features serde
cargo fmt --all

# Check no_std compatibility
cargo check --target riscv32imac-unknown-none-elf --no-default-features
cargo check --target riscv64imac-unknown-none-elf --no-default-features

# Run Ethereum state tests
cargo run -p revme statetest legacytests/Cancun/GeneralStateTests
```

### Test Scripts
```bash
# Download and run ethereum tests
./scripts/run-tests.sh

# Clean test fixtures and re-run
./scripts/run-tests.sh clean

# Run with specific profile
./scripts/run-tests.sh release
```

## Architecture

The workspace consists of these core crates:

- **revm**: Main crate that re-exports all others
- **revm-primitives**: Constants, primitive types, and core data structures
- **revm-interpreter**: EVM opcode implementations and execution engine
- **revm-context**: Execution context, environment, and journaled state
- **revm-handler**: Execution flow control and call frame management
- **revm-database**: State database traits and implementations
- **revm-precompile**: Ethereum precompiled contracts
- **revm-inspector**: Tracing and debugging framework
- **op-revm**: Example of custom EVM variant (Optimism)

### Key Design Patterns

1. **Trait-based Architecture**: Core functionality is defined through traits, allowing custom implementations
2. **Handler Pattern**: Execution flow is controlled through customizable handlers
3. **no_std Support**: All core crates support no_std environments
4. **Feature Flags**: Extensive use of feature flags for optional functionality

### Important Interfaces

1. **Database Trait** (`revm-database`): Defines how state is accessed
2. **Inspector Trait** (`revm-inspector`): Hooks for transaction tracing
3. **Handler Interface** (`revm-handler`): Customizable execution logic
4. **Context** (`revm-context`): Manages execution state and environment

## Current Development Context

When working on the `frame_stack` branch, note that significant refactoring is happening around:
- Frame and FrameData structures (moved from handler to context)
- Execution loop simplification
- Inspector trait cleanup

## Testing Strategy

1. Unit tests in each crate
2. Integration tests using Ethereum official test suite
3. Example projects demonstrating features
4. Benchmarking with CodSpeed

When adding new features:
- Ensure no_std compatibility
- Add appropriate feature flags
- Include tests for new functionality
- Update relevant examples if needed
