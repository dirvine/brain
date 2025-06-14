# NEAT Educational Platform GUI

An interactive desktop application built with Tauri that provides a visual interface for the NEAT educational mathematics platform. This application allows students to solve mathematical problems while visualizing the neural networks that power the AI reasoning.

## Features

### 🎯 Interactive Problem Solving
- **Multiple Mathematical Domains**: Arithmetic, Algebra, Calculus, Trigonometry, Statistics, and Discrete Mathematics
- **Adaptive Difficulty**: Three difficulty levels (Easy, Medium, Hard) that adjust to student performance
- **Real-time Feedback**: Immediate validation and step-by-step explanations for each solution
- **Hint System**: Contextual hints to guide students through challenging problems

### 🧠 Neural Network Visualization
- **Real-time Network Display**: See the actual neural networks that solve mathematical problems
- **Interactive Visualization**: Explore network topology with nodes and weighted connections
- **Performance Metrics**: Live display of accuracy, efficiency, and complexity statistics
- **Evolution Insights**: Understand how NEAT evolves network structures for different mathematical domains

### 📚 Educational Features
- **Step-by-Step Solutions**: Detailed breakdowns of problem-solving processes
- **Mathematical Rendering**: Beautiful display of mathematical expressions and equations
- **Progress Tracking**: Monitor student performance and learning progression
- **Adaptive Learning**: System adapts to individual student needs and skill levels

## Technology Stack

### Frontend
- **Tauri**: Cross-platform desktop application framework
- **TypeScript**: Type-safe JavaScript with modern ES6+ features
- **Vite**: Fast build tool and development server
- **Vis-Network**: Interactive network visualization library
- **KaTeX**: Beautiful mathematical expression rendering

### Backend
- **Rust**: High-performance systems programming language
- **NEAT Library**: Complete implementation of NeuroEvolution of Augmenting Topologies
- **Educational Modules**: Adaptive tutoring, problem generation, and assessment engines
- **Mathematical Engines**: Specialized modules for calculus, statistics, and discrete mathematics

## Prerequisites

- **Rust** 1.75+ with Cargo
- **Node.js** 18+ with npm
- **Tauri CLI** (will be installed automatically)

## Installation and Setup

### 1. Install Dependencies

```bash
# Navigate to the GUI directory
cd gui/neat-edu-gui

# Install Node.js dependencies
npm install

# The Rust dependencies will be installed automatically by Tauri
```

### 2. Development Mode

```bash
# Start the development server
npm run dev

# This will:
# - Start the Vite development server
# - Compile the Rust backend
# - Launch the Tauri application
```

### 3. Production Build

```bash
# Build for production
npm run build

# Create distributable packages
npm run tauri build
```

## Usage Guide

### Getting Started

1. **Launch the Application**: Run `npm run dev` or use the built executable
2. **Select a Topic**: Choose from Arithmetic, Algebra, Calculus, Trigonometry, Statistics, or Discrete Math
3. **Adjust Difficulty**: Use the slider to set Easy, Medium, or Hard difficulty
4. **Generate Problem**: Click "Generate New Problem" to get a mathematical challenge
5. **Solve and Learn**: Enter your answer and receive immediate feedback with explanations

### Understanding the Neural Network Visualization

The right panel shows the actual neural network that solves each mathematical problem:

- **Blue Nodes**: Input neurons that receive the problem data
- **Purple Nodes**: Hidden neurons that process and transform information
- **Green Nodes**: Output neurons that produce the solution
- **Green Edges**: Positive weighted connections
- **Red Edges**: Negative weighted connections
- **Edge Thickness**: Represents the strength of the connection weight

### Performance Metrics

- **Accuracy**: Percentage of problems the network solves correctly
- **Efficiency**: How quickly the network processes problems
- **Complexity**: Number of problems used to train the network

## Educational Value

### For Students
- **Visual Learning**: See how AI actually "thinks" about mathematical problems
- **Step-by-Step Guidance**: Understand the reasoning process behind solutions
- **Adaptive Challenge**: Problems adjust to maintain optimal learning difficulty
- **Immediate Feedback**: Learn from mistakes with detailed explanations

### For Educators
- **Insight into AI**: Demonstrate how machine learning approaches mathematical reasoning
- **Curriculum Support**: Covers multiple mathematical domains with aligned difficulty levels
- **Progress Monitoring**: Track student performance across different topics
- **STEM Integration**: Bridge mathematics and computer science concepts

### For Researchers
- **Algorithm Visualization**: See NEAT evolution in action
- **Performance Analysis**: Study how network topology affects mathematical reasoning
- **Educational Data**: Collect data on learning patterns and problem-solving strategies
- **AI Transparency**: Understand how evolved networks develop mathematical intuition

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Tauri IPC     │    │   Rust Backend  │
│   (TypeScript)  │◄──►│   (JSON API)    │◄──►│   (NEAT Core)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                                                              │
├─ Network Visualization                      ├─ Problem Generation
├─ Problem Interface                          ├─ Solution Validation  
├─ Student Interaction                        ├─ Network Evolution
└─ Progress Display                           └─ Educational Engines
```

### Key Components

1. **Problem Generator**: Creates mathematical problems adapted to student level
2. **Assessment Engine**: Evaluates student responses and provides feedback
3. **Explanation Engine**: Generates step-by-step solution explanations
4. **Network Visualizer**: Renders real-time neural network topology and activity
5. **Adaptive Tutor**: Adjusts difficulty and provides personalized learning paths

## Customization and Extension

### Adding New Mathematical Domains

1. **Backend**: Implement new module types in the NEAT library
2. **Problem Generation**: Add problem templates for the new domain
3. **Frontend**: Add topic buttons and visualization support
4. **Evaluation**: Create domain-specific assessment criteria

### Enhancing Visualizations

- **3D Networks**: Upgrade to three-dimensional network visualization
- **Animation**: Add animated network activation during problem solving
- **Statistics**: Include detailed performance graphs and learning curves
- **Comparison**: Side-by-side network comparison for different approaches

## Troubleshooting

### Common Issues

**Application won't start**
- Ensure Rust and Node.js are properly installed
- Check that all dependencies are installed with `npm install`
- Verify the NEAT library builds correctly in the parent directory

**Network visualization not showing**
- Check browser console for JavaScript errors
- Ensure vis-network library is properly loaded
- Verify that the backend is generating network data correctly

**Mathematical expressions not rendering**
- Confirm KaTeX is loaded and available
- Check for syntax errors in mathematical expressions
- Ensure proper escaping of special characters

### Development Tips

- Use browser developer tools to debug frontend issues
- Check Tauri logs for backend errors and API communication
- Test mathematical engines independently before GUI integration
- Use console logging to trace problem generation and network creation

## Contributing

We welcome contributions to improve the educational platform:

1. **Bug Reports**: Submit detailed issue reports with reproduction steps
2. **Feature Requests**: Suggest new educational features or mathematical domains
3. **Code Contributions**: Implement new visualizations, problem types, or UI improvements
4. **Educational Content**: Add problem templates, explanations, or curriculum alignment

## License

This project is licensed under the MIT License - see the main project LICENSE file for details.

## Acknowledgments

- **NEAT Algorithm**: Kenneth O. Stanley and Risto Miikkulainen
- **Tauri Framework**: The Tauri team for the excellent cross-platform framework
- **Educational Research**: Mathematics education and AI visualization research communities
- **Open Source Libraries**: vis-network, KaTeX, and the broader JavaScript ecosystem