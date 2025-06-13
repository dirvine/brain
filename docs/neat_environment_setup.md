# NEAT Development Environment Setup Guide

## Overview

This guide provides step-by-step instructions for setting up a complete development environment for the NEAT Fashion-MNIST project. It covers all required tools, dependencies, and configurations needed for efficient development.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10/11 with WSL2
- **CPU**: 4-core processor (8+ cores recommended for parallel processing)
- **RAM**: 8GB minimum (16GB+ recommended for large populations)
- **Storage**: 10GB free space (SSD recommended)
- **Network**: Internet connection for downloading dependencies and datasets

### Recommended Specifications
- **CPU**: 8-core processor with AVX2 support
- **RAM**: 32GB for optimal performance
- **Storage**: NVMe SSD with 50GB+ free space
- **GPU**: Optional, for future extensions

## Quick Setup Script

For experienced users, run this automated setup script:

```bash
#!/bin/bash
# Quick setup for NEAT development environment

set -e

echo "🦀 Setting up NEAT Fashion-MNIST development environment..."

# Install Rust
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Install Python
if ! command -v python3 &> /dev/null; then
    echo "Please install Python 3.8+ manually"
    exit 1
fi

# Install development tools
echo "Installing Rust development tools..."
rustup component add rustfmt clippy
cargo install cargo-watch cargo-tarpaulin cargo-audit cargo-outdated

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install --user datasets evaluate torch transformers numpy pandas

# Verify installation
echo "✅ Setup complete!"
echo "Rust version: $(rustc --version)"
echo "Python version: $(python3 --version)"

echo "🚀 Ready to start development!"
```

## Detailed Setup Instructions

### Step 1: Rust Installation and Configuration

#### Installing Rust
```bash
# Install Rust using rustup (official installer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Follow the prompts and choose default installation
# Restart your terminal or run:
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

#### Rust Component Setup
```bash
# Install essential components
rustup component add rustfmt    # Code formatting
rustup component add clippy     # Linting and suggestions
rustup component add rls        # Language server (if not using rust-analyzer)

# Install additional targets if needed
rustup target add x86_64-unknown-linux-gnu
rustup target add x86_64-pc-windows-gnu  # For Windows compatibility
```

#### Cargo Tools Installation
```bash
# Essential development tools
cargo install cargo-watch      # Auto-rebuild on file changes
cargo install cargo-tarpaulin  # Code coverage analysis
cargo install cargo-audit      # Security vulnerability scanning
cargo install cargo-outdated   # Check for outdated dependencies
cargo install cargo-tree       # Visualize dependency trees

# Performance and profiling tools
cargo install flamegraph       # Performance profiling
cargo install cargo-bloat      # Binary size analysis
cargo install criterion        # Benchmarking (if not in Cargo.toml)

# Optional but useful tools
cargo install cargo-expand     # Macro expansion
cargo install cargo-geiger     # Unsafe code detection
cargo install cargo-deny       # License and dependency checking
```

### Step 2: Python Environment Setup

#### Python Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS (using Homebrew)
brew install python3

# Windows (using Chocolatey)
choco install python3

# Verify installation
python3 --version
pip3 --version
```

#### Python Virtual Environment
```bash
# Create virtual environment for the project
python3 -m venv neat-env

# Activate virtual environment
# Linux/macOS:
source neat-env/bin/activate
# Windows:
neat-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Python Dependencies
```bash
# Install HuggingFace and ML libraries
pip install datasets>=2.14.0
pip install evaluate>=0.4.0
pip install torch>=2.0.0
pip install transformers>=4.21.0
pip install numpy>=1.21.0
pip install pandas>=1.5.0

# Development and analysis tools
pip install jupyter>=1.0.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install scikit-learn>=1.1.0

# Verify installation
python3 -c "import datasets, evaluate, torch; print('✅ Python dependencies installed')"
```

### Step 3: IDE and Editor Setup

#### Visual Studio Code (Recommended)

**Installation:**
```bash
# Ubuntu/Debian
sudo snap install code --classic

# macOS
brew install --cask visual-studio-code

# Windows
# Download from https://code.visualstudio.com/
```

**Essential Extensions:**
```bash
# Install via command line
code --install-extension rust-lang.rust-analyzer
code --install-extension ms-python.python
code --install-extension ms-vscode.cpptools
code --install-extension vadimcn.vscode-lldb
code --install-extension tamasfe.even-better-toml
code --install-extension serayuzgur.crates
```

**VS Code Configuration:**
Create `.vscode/settings.json`:
```json
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": "all",
    "rust-analyzer.procMacro.enable": true,
    "rust-analyzer.completion.addCallArgumentSnippets": false,
    
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    },
    
    "[python]": {
        "editor.defaultFormatter": "ms-python.python"
    },
    
    "python.defaultInterpreterPath": "./neat-env/bin/python"
}
```

**VS Code Tasks:**
Create `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "cargo check",
            "type": "cargo",
            "command": "check",
            "args": ["--all-features"],
            "group": "build",
            "presentation": {
                "clear": true,
                "panel": "shared"
            },
            "problemMatcher": "$rustc"
        },
        {
            "label": "cargo test",
            "type": "cargo",
            "command": "test",
            "args": ["--all-features", "--", "--nocapture"],
            "group": "test",
            "presentation": {
                "clear": true,
                "panel": "shared"
            }
        },
        {
            "label": "cargo bench",
            "type": "shell",
            "command": "cargo",
            "args": ["bench", "--", "--output-format", "html"],
            "group": "test",
            "presentation": {
                "clear": true,
                "panel": "shared"
            }
        },
        {
            "label": "cargo run example",
            "type": "cargo",
            "command": "run",
            "args": ["--example", "${input:exampleName}"],
            "group": "build"
        }
    ],
    "inputs": [
        {
            "id": "exampleName",
            "description": "Example to run",
            "default": "basic_evolution",
            "type": "promptString"
        }
    ]
}
```

**Launch Configuration:**
Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug unit tests",
            "cargo": {
                "args": ["test", "--no-run", "--lib"],
                "filter": {
                    "name": "neat_fashion_classifier",
                    "kind": "lib"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        },
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug example",
            "cargo": {
                "args": ["build", "--example", "basic_evolution"],
                "filter": {
                    "name": "basic_evolution",
                    "kind": "example"
                }
            },
            "args": [],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

#### Alternative IDEs

**CLion/IntelliJ with Rust Plugin:**
- Install Rust plugin from JetBrains marketplace
- Configure Rust toolchain in settings
- Enable Clippy integration
- Set up run configurations for tests and examples

**Vim/Neovim with Rust Support:**
```vim
" Add to .vimrc or init.vim
Plug 'rust-lang/rust.vim'
Plug 'dense-analysis/ale'
Plug 'neoclide/coc.nvim'

" Configure rust.vim
let g:rustfmt_autosave = 1
let g:rust_clip_command = 'xclip -selection clipboard'

" Configure ALE for Rust
let g:ale_linters = {'rust': ['analyzer']}
let g:ale_fixers = {'rust': ['rustfmt']}
```

### Step 4: Git and Version Control Setup

#### Git Configuration
```bash
# Configure Git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Configure Git preferences
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.editor "code --wait"  # or your preferred editor

# Configure helpful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.cm commit
git config --global alias.lg "log --oneline --graph --decorate --all"
```

#### Git Hooks Setup
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
# Pre-commit hook for Rust projects

set -e

echo "Running pre-commit checks..."

# Format code
cargo fmt --all -- --check
if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
    exit 1
fi

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
if [ $? -ne 0 ]; then
    echo "❌ Clippy check failed. Fix warnings before committing."
    exit 1
fi

# Run tests
cargo test --all-features
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Fix failing tests before committing."
    exit 1
fi

echo "✅ All pre-commit checks passed!"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

### Step 5: Project-Specific Setup

#### Project Initialization
```bash
# Clone or create the project
git clone https://github.com/your-username/neat-fashion-classifier.git
cd neat-fashion-classifier

# Or create new project
cargo new neat-fashion-classifier --lib
cd neat-fashion-classifier
```

#### Cargo Configuration
Create `.cargo/config.toml`:
```toml
[build]
target-dir = "target"

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native"]

[env]
RUST_BACKTRACE = "1"

# Speed up builds during development
[profile.dev]
debug = true
opt-level = 0
overflow-checks = true

[profile.dev.package."*"]
opt-level = 2  # Optimize dependencies even in debug mode

# Optimize for performance in release builds
[profile.release]
debug = false
lto = true
codegen-units = 1
panic = "abort"
opt-level = 3

# Profile for benchmarking
[profile.bench]
debug = true
opt-level = 3
```

#### Rustfmt Configuration
Create `.rustfmt.toml`:
```toml
# Line length and layout
max_width = 100
hard_tabs = false
tab_spaces = 4

# Import organization
imports_layout = "Mixed"
group_imports = "StdExternalCrate"

# Code style
fn_args_layout = "Tall"
brace_style = "SameLineWhere"
control_brace_style = "AlwaysSameLine"
trailing_semicolon = true
trailing_comma = "Vertical"

# Comments and documentation
wrap_comments = true
comment_width = 80
normalize_comments = true
normalize_doc_attributes = true

# String handling
string_lit_normalize = true
format_strings = true

# Misc formatting
merge_derives = true
use_field_init_shorthand = true
use_try_shorthand = true
```

#### Clippy Configuration
Create `.clippy.toml`:
```toml
# Threshold for cognitive complexity
cognitive-complexity-threshold = 30

# Threshold for type complexity
type-complexity-threshold = 250

# Single char lifetime names allowed
single-char-lifetime-names-threshold = 4

# Trivial copy types max size
trivial-copy-size-limit = 64

# Too many arguments threshold
too-many-arguments-threshold = 7

# Too many lines threshold
too-many-lines-threshold = 100
```

Add to `Cargo.toml`:
```toml
[lints.clippy]
# Enable additional lints
pedantic = "warn"
nursery = "warn"
cargo = "warn"

# Allow certain pedantic lints that can be noisy
module_name_repetitions = "allow"
must_use_candidate = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
```

### Step 6: Performance Tools Setup

#### Profiling Tools
```bash
# Install perf (Linux)
sudo apt install linux-tools-common linux-tools-generic

# Install valgrind (memory analysis)
sudo apt install valgrind

# Install heaptrack (heap profiling)
sudo apt install heaptrack

# Configure flamegraph
echo 'kernel.perf_event_paranoid = -1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### Benchmarking Setup
Create `benches/` directory and basic benchmark:
```rust
// benches/neat_benchmarks.rs
use criterion::{criterion_group, criterion_main, Criterion};
use neat_fashion_classifier::*;

fn benchmark_basic_operations(c: &mut Criterion) {
    c.bench_function("genome_creation", |b| {
        b.iter(|| {
            let genome = Genome::new(0, 784, 10);
            genome
        })
    });
}

criterion_group!(benches, benchmark_basic_operations);
criterion_main!(benches);
```

### Step 7: Documentation Setup

#### Documentation Tools
```bash
# Install mdbook for documentation
cargo install mdbook
cargo install mdbook-mermaid  # For diagrams

# Create documentation structure
mdbook init docs --title "NEAT Fashion-MNIST Guide"
```

#### Documentation Configuration
Edit `docs/book.toml`:
```toml
[book]
authors = ["Your Name"]
language = "en"
multilingual = false
src = "src"
title = "NEAT Fashion-MNIST Classifier"
description = "Complete guide for NEAT implementation in Rust"

[preprocessor.mermaid]
command = "mdbook-mermaid"

[output.html]
default-theme = "light"
preferred-dark-theme = "navy"
git-repository-url = "https://github.com/your-username/neat-fashion-classifier"
edit-url-template = "https://github.com/your-username/neat-fashion-classifier/edit/main/docs/{path}"

[output.html.print]
enable = true

[output.html.search]
enable = true
limit-results = 30
teaser-word-count = 30
use-boolean-and = true
boost-title = 2
boost-hierarchy = 1
boost-paragraph = 1
```

### Step 8: Continuous Integration Setup

#### GitHub Actions
Create `.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always

jobs:
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]

    steps:
    - uses: actions/checkout@v3

    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: ${{ matrix.rust }}
        components: rustfmt, clippy
        override: true

    - name: Cache cargo registry
      uses: actions/cache@v3
      with:
        path: ~/.cargo/registry
        key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}

    - name: Cache cargo index
      uses: actions/cache@v3
      with:
        path: ~/.cargo/git
        key: ${{ runner.os }}-cargo-index-${{ hashFiles('**/Cargo.lock') }}

    - name: Cache cargo build
      uses: actions/cache@v3
      with:
        path: target
        key: ${{ runner.os }}-cargo-build-target-${{ hashFiles('**/Cargo.lock') }}

    - name: Check formatting
      run: cargo fmt --all -- --check

    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings

    - name: Run tests
      run: cargo test --all-features --verbose

    - name: Run doc tests
      run: cargo test --doc --all-features

  coverage:
    name: Coverage
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true

    - name: Install tarpaulin
      run: cargo install cargo-tarpaulin

    - name: Generate code coverage
      run: cargo tarpaulin --verbose --all-features --workspace --timeout 120 --out Xml

    - name: Upload to codecov.io
      uses: codecov/codecov-action@v3
      with:
        fail_ci_if_error: true

  benchmark:
    name: Benchmark
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3

    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        override: true

    - name: Run benchmarks
      run: cargo bench -- --output-format html

    - name: Store benchmark result
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: target/criterion
```

### Step 9: Environment Validation

#### Validation Script
Create `scripts/validate_env.sh`:
```bash
#!/bin/bash
# Environment validation script

set -e

echo "🔍 Validating development environment..."

# Check Rust installation
echo "Checking Rust installation..."
rustc --version || { echo "❌ Rust not installed"; exit 1; }
cargo --version || { echo "❌ Cargo not available"; exit 1; }

# Check Rust components
echo "Checking Rust components..."
rustup component list --installed | grep rustfmt || { echo "❌ rustfmt not installed"; exit 1; }
rustup component list --installed | grep clippy || { echo "❌ clippy not installed"; exit 1; }

# Check Python installation
echo "Checking Python installation..."
python3 --version || { echo "❌ Python 3 not installed"; exit 1; }
pip3 --version || { echo "❌ pip3 not available"; exit 1; }

# Check Python packages
echo "Checking Python packages..."
python3 -c "import datasets" || { echo "❌ datasets package not installed"; exit 1; }
python3 -c "import evaluate" || { echo "❌ evaluate package not installed"; exit 1; }
python3 -c "import torch" || { echo "❌ torch package not installed"; exit 1; }

# Check development tools
echo "Checking development tools..."
cargo-watch --version || { echo "⚠️  cargo-watch not installed (optional)"; }
cargo-tarpaulin --version || { echo "⚠️  cargo-tarpaulin not installed (optional)"; }

# Check Git configuration
echo "Checking Git configuration..."
git config user.name || { echo "⚠️  Git user.name not configured"; }
git config user.email || { echo "⚠️  Git user.email not configured"; }

# Test basic functionality
echo "Testing basic functionality..."
cargo check --version || { echo "❌ cargo check failed"; exit 1; }

# Test HuggingFace integration
echo "Testing HuggingFace integration..."
python3 -c "
from datasets import load_dataset
print('✅ HuggingFace datasets working')
"

echo "✅ Environment validation complete!"
echo "🚀 Ready for NEAT development!"
```

Make it executable and run:
```bash
chmod +x scripts/validate_env.sh
./scripts/validate_env.sh
```

### Step 10: First Build Test

#### Create Test Project
```bash
# Initialize new Rust project
cargo new neat-test --lib
cd neat-test

# Add basic dependencies to Cargo.toml
cat >> Cargo.toml << EOF

[dependencies]
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
rand = "0.8"
EOF

# Create simple test
cat > src/lib.rs << 'EOF'
//! Test NEAT environment setup

use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TestGenome {
    pub id: usize,
    pub weights: Vec<f64>,
}

impl TestGenome {
    pub fn new(id: usize, size: usize) -> Self {
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        let weights = (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        Self { id, weights }
    }
    
    pub fn activate(&self, inputs: &[f64]) -> f64 {
        let input_array = Array1::from_vec(inputs.to_vec());
        let weight_array = Array1::from_vec(self.weights.clone());
        input_array.dot(&weight_array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genome_creation() {
        let genome = TestGenome::new(0, 10);
        assert_eq!(genome.id, 0);
        assert_eq!(genome.weights.len(), 10);
    }

    #[test]
    fn test_activation() {
        let genome = TestGenome::new(0, 3);
        let inputs = vec![1.0, 0.5, -0.5];
        let output = genome.activate(&inputs);
        assert!(output.is_finite());
    }
}
EOF

# Test compilation and execution
cargo test --verbose
echo "✅ Test project compiled and ran successfully!"

# Clean up
cd ..
rm -rf neat-test
```

## Troubleshooting

### Common Issues and Solutions

#### Rust Installation Issues
```bash
# Permission denied during installation
sudo chown -R $(whoami) ~/.cargo ~/.rustup

# PATH not updated
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Old Rust version
rustup update stable
```

#### Python Package Issues
```bash
# Permission denied for pip install
python3 -m pip install --user <package>

# Virtual environment activation issues
deactivate  # if already in venv
python3 -m venv neat-env --clear
source neat-env/bin/activate

# Package conflicts
pip install --upgrade pip
pip install --force-reinstall <package>
```

#### Performance Issues
```bash
# Slow compilation
export CARGO_INCREMENTAL=1
export RUSTC_WRAPPER=sccache  # if sccache installed

# Memory issues during compilation
export CARGO_BUILD_JOBS=2  # Reduce parallel jobs

# Linker issues on Linux
sudo apt install build-essential
```

#### IDE Issues
```bash
# VS Code rust-analyzer not working
code --install-extension rust-lang.rust-analyzer --force
# Restart VS Code

# Intellisense slow
# Add to VS Code settings:
"rust-analyzer.cargo.loadOutDirsFromCheck": true
"rust-analyzer.checkOnSave.enable": false  # Temporarily
```

### Performance Optimization Tips

#### Development Build Speed
```toml
# Add to Cargo.toml
[profile.dev.package."*"]
opt-level = 2  # Optimize dependencies

[profile.dev]
debug = 1  # Reduce debug info
```

#### Runtime Performance
```bash
# Use CPU-specific optimizations
export RUSTFLAGS="-C target-cpu=native"

# Enable link-time optimization for release builds
cargo build --release
```

## Next Steps

After completing this setup:

1. **Validate Environment**: Run the validation script to ensure everything works
2. **Clone Project**: Get the NEAT project repository
3. **Run Initial Tests**: Verify the development workflow
4. **Start Development**: Begin with Week 1 tasks from the development plan
5. **Set Up Monitoring**: Configure performance monitoring and logging

## Getting Help

### Resources
- **Rust Documentation**: https://doc.rust-lang.org/
- **Cargo Book**: https://doc.rust-lang.org/cargo/
- **HuggingFace Docs**: https://huggingface.co/docs
- **NEAT Papers**: Original research papers in the project documentation

### Community Support
- **Rust Users Forum**: https://users.rust-lang.org/
- **Rust Discord**: https://discord.gg/rust-lang
- **Stack Overflow**: Tag questions with `rust`, `neat-algorithm`, `machine-learning`

### Project-Specific Help
- Check project README and documentation
- Review existing issues in the repository
- Create new issues for bugs or feature requests
- Join project discussions and code reviews

This comprehensive setup guide ensures you have everything needed to develop the NEAT Fashion-MNIST classifier efficiently and effectively.