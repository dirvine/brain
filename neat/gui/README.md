# NEAT Educational GUI

This directory contains the interactive desktop application for the NEAT Educational Platform, built with Tauri and TypeScript.

## 🚀 Quick Start

### Prerequisites

Make sure you have the following installed:
- **Node.js** 18+ with npm
- **Rust** (latest stable version)
- **Git** (for cloning the repository)

### Running the Application

1. **Navigate to the GUI directory:**
   ```bash
   cd gui/neat-edu-gui
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```
   This will install all frontend dependencies including Tauri CLI, TypeScript, Vite, and visualization libraries.

3. **Launch the application:**
   ```bash
   npm run tauri:dev
   ```
   
   **Alternative (browser preview only):**
   ```bash
   npm run dev
   ```
   
   **First Run Notes:**
   - The first launch may take 5-10 minutes as it compiles the entire Rust backend
   - This includes building the NEAT library and all mathematical modules
   - Subsequent runs will be much faster (30-60 seconds)
   - Use `npm run tauri:dev` for the full desktop experience with neural network backend
   - Use `npm run dev` for frontend-only browser preview with demo mode

4. **The application will open in a desktop window showing:**
   - Interactive mathematical problem solver (left panel)
   - Real-time neural network visualization (right panel)

## 🎯 What You'll See

### Problem Solving Interface
- **Topic Selection**: Choose from Arithmetic, Algebra, Calculus, Trigonometry, Statistics, Discrete Math
- **Difficulty Slider**: Adjust problem difficulty (Easy/Medium/Hard)
- **Interactive Solving**: Enter answers and get immediate feedback
- **Hints & Explanations**: Step-by-step guidance when needed

### Neural Network Visualization
- **Live Networks**: See the actual neural networks solving problems
- **Performance Metrics**: Accuracy, efficiency, and complexity stats
- **Interactive Exploration**: Click nodes and edges for details
- **Evolution Insights**: Understand how NEAT evolves mathematical reasoning

## 🛠️ Development Commands

```bash
# Full desktop application (with Rust backend)
npm run tauri:dev

# Frontend-only browser preview (demo mode)
npm run dev

# Build for production
npm run build

# Create distributable packages
npm run tauri:build

# Preview production build
npm run preview
```

## 📁 Project Structure

```
neat-edu-gui/
├── src/                    # TypeScript frontend
│   └── main.ts            # Main application logic
├── src-tauri/             # Rust backend
│   ├── src/main.rs        # Tauri backend with NEAT integration
│   └── Cargo.toml         # Rust dependencies
├── index.html             # Main HTML interface
├── package.json           # Node.js dependencies
└── README.md              # Detailed documentation
```

## 🔧 Troubleshooting

### Common Issues

**"tauri command not found"**
- Tauri CLI is installed automatically with `npm install`
- Try: `npx tauri --version`

**Long compilation times**
- First run compiles the entire Rust ecosystem
- Subsequent runs use cached builds
- Use `npm run dev` for development (faster rebuilds)

**Application won't start**
- Ensure Node.js 18+ is installed: `node --version`
- Ensure Rust is installed: `rustc --version`
- Clear npm cache: `npm cache clean --force`
- Reinstall: `rm -rf node_modules && npm install`

**Network visualization not showing**
- Check browser console in the developer tools
- Ensure vis-network library loaded correctly
- Verify backend is generating network data

### Platform-Specific Notes

**macOS**
- May need to allow the app in Security & Privacy settings
- Ensure Xcode command line tools are installed

**Windows**
- Windows Defender might scan the first build
- Ensure Visual Studio Build Tools are installed

**Linux**
- May need additional development packages
- Install webkit2gtk and other dependencies as needed

## 📚 Educational Features

### For Students
- **Visual AI Learning**: See how neural networks solve math problems
- **Step-by-Step Guidance**: Detailed explanations for each solution
- **Adaptive Difficulty**: Problems adjust to your skill level
- **Multi-Domain Learning**: Cover multiple areas of mathematics

### For Educators
- **AI Transparency**: Show students how machine learning works
- **Curriculum Support**: Aligned with mathematical learning objectives
- **Progress Tracking**: Monitor student performance and growth
- **STEM Integration**: Bridge math and computer science concepts

### For Researchers
- **Algorithm Visualization**: Study NEAT evolution in real-time
- **Educational Data**: Collect learning pattern analytics
- **AI Explainability**: Research transparent AI in education

## 🔗 Related Documentation

- **[Detailed GUI Documentation](neat-edu-gui/README.md)**: Complete setup and usage guide
- **[Educational Platform Overview](../docs/educational_gui_overview.md)**: Architecture and design philosophy
- **[NEAT Core Documentation](../README.md)**: Main project documentation
- **[Phase 6 Advanced Math](../docs/phase6_advanced_mathematics.md)**: Mathematical domain coverage

## 🤝 Contributing

Interested in improving the educational platform? See the main project README for contribution guidelines. The GUI particularly welcomes:

- **UI/UX Improvements**: Better educational interfaces
- **Visualization Enhancements**: More engaging network displays
- **Educational Content**: New problem types and explanations
- **Accessibility Features**: Support for diverse learning needs

## 📄 License

This project is licensed under the MIT License - see the main project LICENSE file for details.

---

**🧠 Experience the future of AI-powered education - where students learn alongside evolving neural networks!**