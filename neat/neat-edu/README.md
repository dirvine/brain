# 🧠 NEAT Educational Platform

[![Crates.io](https://img.shields.io/crates/v/neat-edu.svg)](https://crates.io/crates/neat-edu)
[![Documentation](https://docs.rs/neat-edu/badge.svg)](https://docs.rs/neat-edu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive educational platform that uses **NEAT (NeuroEvolution of Augmenting Topologies)** neural networks to teach mathematics with real-time network visualization.

## ✨ Features

- 🎯 **Interactive Mathematical Problem Solving** across multiple domains
- 🧠 **Real-time Neural Network Visualization** showing AI reasoning
- 📊 **Performance Metrics** with accuracy, efficiency, and complexity analysis
- 🎲 **Randomized Problem Generation** for unlimited practice
- 🎨 **Beautiful Desktop GUI** built with Tauri and TypeScript
- 📚 **Multiple Mathematical Topics**: Arithmetic, Algebra, Calculus, Trigonometry, Statistics, Discrete Math

## 🚀 Quick Start

### Installation

```bash
cargo install neat-edu
```

### Running the Application

```bash
neat-edu
```

This will launch the desktop GUI application where you can:

1. **Select a mathematical topic** (Arithmetic, Algebra, Calculus, etc.)
2. **Adjust difficulty level** (Easy/Medium/Hard)
3. **Generate random problems** and solve them interactively
4. **Visualize neural networks** processing your solutions in real-time
5. **Track performance metrics** and learning progress

## 🧮 Mathematical Domains

### 🔢 Arithmetic
- **Easy**: Random addition problems (1-20)
- **Medium**: Random multiplication problems (10-50 × 10-20)
- **Hard**: Random division with decimal precision

### 🧮 Algebra
- **Easy**: Linear equations like `x + 7 = 15`
- **Medium**: Linear equations like `3x - 4 = 17`
- **Hard**: Quadratic equations like `x² - 5x + 6 = 0`

### 📈 Calculus
- **Easy**: Polynomial derivatives like `d/dx(x³) = 3x²`
- **Medium**: Complex derivatives with multiple terms
- **Hard**: Integration problems with step-by-step solutions

### 📐 Trigonometry
- **Easy**: Basic trig values like `sin(30°) = 0.5`
- **Medium**: Trigonometric functions and identities
- **Hard**: Solving trigonometric equations

### 📊 Statistics
- **Easy**: Mean calculations with random datasets
- **Medium**: Median finding with shuffled lists
- **Hard**: Standard deviation calculations

### 🎲 Discrete Mathematics
- **Easy**: Factorial and permutation problems
- **Medium**: Combination calculations like `C(n,r)`
- **Hard**: Set theory and combinatorial reasoning

## 🧠 Neural Network Visualization

Watch as NEAT neural networks evolve and adapt to solve mathematical problems:

- **Input Layer**: Problem data encoding
- **Hidden Layers**: Mathematical reasoning and pattern recognition
- **Output Layer**: Solution generation and confidence
- **Real-time Metrics**: Accuracy, efficiency, and network complexity

## 🛠️ Development

### Prerequisites

- Rust 1.75+
- Node.js 18+ with npm
- Tauri CLI

### Building from Source

```bash
git clone https://github.com/your-username/neat-edu
cd neat-edu
npm install
cargo tauri dev
```

### Project Structure

```
neat-edu/
├── src/
│   ├── main.rs              # Main application entry point
│   ├── problem_generator.rs # Randomized math problem generation
│   └── network_visualizer.rs # Neural network visualization
├── src/                     # Frontend TypeScript/HTML
├── Cargo.toml              # Rust dependencies and metadata
└── package.json            # Node.js dependencies
```

## 🎯 Educational Value

This platform demonstrates:

- **AI-Powered Learning**: How neural networks can assist in education
- **Mathematical Reasoning**: Step-by-step problem-solving approaches
- **Visual Learning**: Network topology and mathematical pattern recognition
- **Adaptive Difficulty**: Problems that scale with student ability
- **Interactive Feedback**: Immediate validation and explanation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for contribution:

- **New Mathematical Domains**: Geometry, Number Theory, Linear Algebra
- **Enhanced Visualizations**: 3D networks, animation, interactive exploration
- **Educational Features**: Progress tracking, curriculum integration
- **Performance Optimizations**: Faster problem generation, better validation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- **NEAT Algorithm**: Stanley & Miikkulainen for the foundational NEAT algorithm
- **Tauri Framework**: For enabling cross-platform desktop applications
- **vis-network**: For beautiful network visualization capabilities
- **Rust Community**: For the exceptional tools and ecosystem

## 🔗 Links

- [Documentation](https://docs.rs/neat-edu)
- [Crates.io](https://crates.io/crates/neat-edu)
- [Issues](https://github.com/your-username/neat-edu/issues)
- [Discussions](https://github.com/your-username/neat-edu/discussions)

---

**🚀 Transform mathematical learning with the power of evolving neural networks!**