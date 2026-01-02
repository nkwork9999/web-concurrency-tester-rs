# web-concurrency-tester-rs 🦀🕸️

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Rust](https://img.shields.io/badge/built_with-Rust-orange.svg)](https://www.rust-lang.org)

**web-concurrency-tester-rs** is a deterministic concurrency testing tool for Web/JavaScript environments, written entirely in Rust.

It helps developers find hard-to-reproduce race conditions in DOM operations by using advanced scheduling algorithms like **Shuttle**-style scheduling, **DPOR** (Dynamic Partial Order Reduction), and **PCT** (Probabilistic Concurrency Testing).

## ✨ Features

- **Deterministic Execution**: Controls the order of async operations and events to reliably reproduce bugs.
- **Advanced Race Detection**:
  - **FastTrack / Vector Clocks**: Efficiently tracks _happens-before_ relationships.
  - **DPOR**: Prunes the search space to test only meaningful interleavings.
  - **PCT**: Mathematically guaranteed bug detection probabilities for depth-bounded races.
- **Embedded JS Engine**: Powered by `boa_engine` and `oxc_parser` for direct analysis and execution in Rust.

## 🚀 Usage

### 1. Prerequisites

Ensure you have Rust and Cargo installed.

```bash
git clone [https://github.com/nkwork9999/web-concurrency-tester-rs.git](https://github.com/nkwork9999/web-concurrency-tester-rs.git)
cd web-concurrency-tester-rs

```

### 2. Create a Test Case

Create an HTML file (e.g., `test.html`) containing the JavaScript logic you want to verify.

```html
<button onclick="inc()">Increment</button>
<script>
  let count = 0;

  // A function with a potential race condition
  async function inc() {
    let temp = count;
    // Context switch point
    await new Promise((r) => setTimeout(r, 10));
    count = temp + 1;
  }
</script>
```

### 3. Run the Test

Run the tool targeting your HTML file. Using `--release` mode is recommended for performance.

```bash
cargo run --release -- test.html

```

### Output Example

The tool explores multiple execution schedules and reports any detected races:

```text
  DETECTED RACES:
  [Write-Write] 'count': Task 0 vs Task 1
  ...

```

## 📄 License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.
See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
