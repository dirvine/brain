import { invoke } from '@tauri-apps/api/core';
import { DataSet, Network } from 'vis-network/standalone/esm/vis-network';

// Check if we're running in Tauri context
const isTauri = typeof window !== 'undefined' && window.__TAURI_INTERNALS__;

// Type definitions matching Rust backend
interface NetworkVisualization {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  metrics: NetworkMetrics;
}

interface NetworkNode {
  id: string;
  label: string;
  node_type: string;
  value: number;
  x: number;
  y: number;
  color: string;
}

interface NetworkEdge {
  from: string;
  to: string;
  weight: number;
  color: string;
  width: number;
}

interface NetworkMetrics {
  accuracy: number;
  efficiency: number;
  complexity: number;
  nodes_count: number;
  edges_count: number;
}

interface ProblemRequest {
  topic: string;
  difficulty: string;
  problem_type: string;
}

interface ProblemResponse {
  problem_text: string;
  expected_answer: string;
  explanation: string;
  hints: string[];
  network_data: NetworkVisualization;
}

interface SolutionRequest {
  problem: string;
  student_answer: string;
}

interface SolutionResponse {
  answer: string;
  explanation: string;
  steps: string[];
  network_data: NetworkVisualization;
  correct: boolean;
}

interface EducationalProblemData {
  id: string;
  problem_type: string;
  statement: string;
  hint?: string;
  expected_answer: string;
  difficulty: string;
  topic: string;
}

interface SolutionRequest {
  problem_id: string;
  student_answer: string;
  student_id?: string;
}


// Global state
let currentProblem: EducationalProblemData | null = null;
let currentNetwork: Network | null = null;
let selectedTopic = 'arithmetic';
let selectedDifficulty = 'medium';

// Initialize the application
async function initializeApp() {
  try {
    console.log('Initializing educational platform...');
    
    // Check if we're in Tauri context
    if (!isTauri) {
      console.warn('Running in browser mode - Tauri APIs not available');
      showError('This application requires the Tauri desktop environment. Please run with: npm run tauri dev');
      setupEventListeners();
      updateUI();
      return;
    }
    
    await invoke('initialize_educational_platform');
    console.log('Platform initialized successfully!');
    
    setupEventListeners();
    updateUI();
  } catch (error) {
    console.error('Failed to initialize platform:', error);
    showError('Failed to initialize educational platform: ' + error);
  }
}

function setupEventListeners() {
  // Topic selection
  document.querySelectorAll('.topic-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.target as HTMLButtonElement;
      const topic = target.getAttribute('data-topic');
      if (topic) {
        selectTopic(topic);
      }
    });
  });

  // Difficulty slider
  const difficultySlider = document.getElementById('difficulty') as HTMLInputElement;
  difficultySlider.addEventListener('input', (e) => {
    const target = e.target as HTMLInputElement;
    const levels = ['easy', 'medium', 'hard'];
    selectedDifficulty = levels[parseInt(target.value) - 1];
  });

  // Generate problem button
  document.getElementById('generate-problem')?.addEventListener('click', generateProblem);

  // Submit answer button
  document.getElementById('submit-answer')?.addEventListener('click', submitAnswer);

  // Show hint button
  document.getElementById('show-hint')?.addEventListener('click', showHint);

  // Answer input enter key
  document.getElementById('answer-input')?.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      submitAnswer();
    }
  });
}

function selectTopic(topic: string) {
  selectedTopic = topic;
  
  // Update UI
  document.querySelectorAll('.topic-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  document.querySelector(`[data-topic="${topic}"]`)?.classList.add('active');
}

async function generateProblem() {
  try {
    showLoading('problem-display', 'Generating new problem...');
    hideElements(['answer-input', 'submit-answer', 'show-hint', 'hint-panel', 'explanation-panel']);

    // Check if Tauri is available
    if (!isTauri) {
      showDemoMode();
      return;
    }

    const request: ProblemRequest = {
      topic: selectedTopic,
      difficulty: selectedDifficulty,
      problem_type: 'standard'
    };

    console.log('Generating problem:', request);
    const response: ProblemResponse = await invoke('generate_problem', { request });
    console.log('Problem generated:', response);

    // Store the current problem for answer checking
    currentProblem = {
      id: 'generated-' + Date.now(),
      problem_type: 'standard',
      topic: selectedTopic,
      difficulty: selectedDifficulty,
      statement: response.problem_text,
      expected_solution: response.expected_answer,
      hint: response.hints[0] || 'No hint available'
    };
    
    displayGeneratedProblem(response.problem_text, response.explanation);
    visualizeNetwork(response.network_data);
    updatePerformanceStats(response.network_data.metrics);

    // Show input elements
    showElements(['answer-input', 'submit-answer', 'show-hint']);
    
    // Clear previous answer
    const answerInput = document.getElementById('answer-input') as HTMLInputElement;
    answerInput.value = '';
    answerInput.focus();

  } catch (error) {
    console.error('Failed to generate problem:', error);
    showError('Failed to generate problem: ' + error);
  }
}

function displayProblem(problem: EducationalProblemData) {
  const problemDisplay = document.getElementById('problem-display');
  if (problemDisplay) {
    problemDisplay.innerHTML = `
      <div class="problem-text">
        <strong>${problem.topic} Problem (${problem.difficulty}):</strong><br>
        <div style="margin-top: 1rem; font-size: 1.2rem;">${problem.statement}</div>
      </div>
    `;
  }
}

function displayGeneratedProblem(problemText: string, explanation: string) {
  const problemDisplay = document.getElementById('problem-display');
  if (problemDisplay) {
    problemDisplay.innerHTML = `
      <div class="problem-text">
        <strong>${selectedTopic.charAt(0).toUpperCase() + selectedTopic.slice(1)} Problem (${selectedDifficulty}):</strong><br>
        <div style="margin-top: 1rem; font-size: 1.2rem;">${problemText}</div>
        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">${explanation}</div>
      </div>
    `;
  }
}

async function submitAnswer() {
  if (!currentProblem) {
    showError('No problem to solve!');
    return;
  }

  const answerInput = document.getElementById('answer-input') as HTMLInputElement;
  const answer = answerInput.value.trim();

  if (!answer) {
    showError('Please enter an answer!');
    return;
  }

  // Check if Tauri is available
  if (!isTauri) {
    showDemoSolution(answer);
    return;
  }

  try {
    const request: SolutionRequest = {
      problem: currentProblem.statement,
      student_answer: answer
    };
    
    console.log('Submitting answer:', request);
    const response: SolutionResponse = await invoke('solve_problem', { request });
    console.log('Solution response:', response);

    displaySolution(response);
    visualizeNetwork(response.network_data);

  } catch (error) {
    console.error('Failed to submit answer:', error);
    showError('Failed to submit answer: ' + error);
  }
}

function displaySolution(response: SolutionResponse) {
  const explanationPanel = document.getElementById('explanation-panel');
  if (explanationPanel) {
    const statusClass = response.correct ? 'success' : 'error';
    const statusIcon = response.correct ? '✅' : '❌';
    
    explanationPanel.innerHTML = `
      <div class="${statusClass}" style="padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        ${statusIcon} <strong>${response.correct ? 'Correct!' : 'Incorrect'}</strong><br>
        ${response.explanation}
      </div>
      
      <div style="margin-bottom: 1rem;">
        <strong>Step-by-Step Solution:</strong>
        <ol class="step-list">
          ${response.steps.map(step => `<li>${step}</li>`).join('')}
        </ol>
      </div>
      
      <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
        <strong>Your Answer:</strong> ${response.answer}<br>
        <strong>Neural Network Analysis:</strong> The network processed this problem with high confidence.
      </div>
    `;
    
    explanationPanel.style.display = 'block';
  }

  // Disable submit button
  const submitBtn = document.getElementById('submit-answer') as HTMLButtonElement;
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = response.correct ? '✅ Correct!' : '❌ Try Again';
    
    // Re-enable after 3 seconds
    setTimeout(() => {
      submitBtn.disabled = false;
      submitBtn.textContent = '✓ Submit Answer';
    }, 3000);
  }
}

function showHint() {
  if (!currentProblem?.hint) {
    showError('No hint available for this problem.');
    return;
  }

  const hintPanel = document.getElementById('hint-panel');
  if (hintPanel) {
    hintPanel.textContent = currentProblem.hint;
    hintPanel.style.display = 'block';
  }

  // Disable hint button
  const hintBtn = document.getElementById('show-hint') as HTMLButtonElement;
  if (hintBtn) {
    hintBtn.disabled = true;
    hintBtn.textContent = '💡 Hint Shown';
  }
}

function visualizeNetwork(networkViz: NetworkVisualization) {
  const container = document.getElementById('network-container');
  if (!container) return;

  // Clear previous network
  if (currentNetwork) {
    currentNetwork.destroy();
  }

  // Prepare data for vis-network
  const nodes = new DataSet(networkViz.nodes.map(node => ({
    id: node.id,
    label: node.label,
    color: node.color || getNodeColor(node.node_type),
    size: getNodeSize(node.node_type, node.value),
    font: { size: 12, color: '#333' },
    x: node.x,
    y: node.y,
    physics: false
  })));

  const edges = new DataSet(networkViz.edges.map(edge => ({
    from: edge.from,
    to: edge.to,
    width: edge.width || Math.abs(edge.weight) * 3,
    color: edge.color || (edge.weight > 0 ? '#4ade80' : '#ef4444'),
    arrows: { to: { enabled: true, scaleFactor: 0.5 } }
  })));

  // Create network
  const data = { nodes, edges };
  const options = {
    layout: {
      hierarchical: false
    },
    physics: {
      enabled: false
    },
    interaction: {
      dragNodes: false,
      zoomView: true
    },
    nodes: {
      shape: 'circle',
      borderWidth: 2,
      borderColor: '#333'
    },
    edges: {
      smooth: {
        type: 'continuous'
      }
    }
  };

  currentNetwork = new Network(container, data, options);

  // Update network info
  updateNetworkInfo(networkViz);
}

function getNodeColor(nodeType: string): string {
  switch (nodeType) {
    case 'input': return '#60a5fa';
    case 'output': return '#34d399';
    case 'hidden': return '#a78bfa';
    default: return '#94a3b8';
  }
}

function getNodeSize(nodeType: string, activation: number): number {
  const baseSize = nodeType === 'hidden' ? 20 : 25;
  return baseSize + Math.abs(activation) * 10;
}

function updateNetworkInfo(networkViz: NetworkVisualization) {
  const infoElement = document.getElementById('network-info');
  if (infoElement) {
    const nodeCount = networkViz.nodes.length;
    const edgeCount = networkViz.edges.length;
    const hiddenNodes = networkViz.nodes.filter(n => n.node_type === 'hidden').length;
    
    infoElement.innerHTML = `
      <strong>Module Type:</strong> ${networkViz.module_type}<br>
      <strong>Network Structure:</strong> ${nodeCount} nodes, ${edgeCount} connections<br>
      <strong>Hidden Layers:</strong> ${hiddenNodes} hidden neurons<br>
      <strong>Evolution:</strong> Evolved topology using NEAT algorithm
    `;
  }
}

function updatePerformanceStats(metrics: NetworkMetrics) {
  const accuracyStat = document.getElementById('accuracy-stat');
  const efficiencyStat = document.getElementById('efficiency-stat');
  const complexityStat = document.getElementById('complexity-stat');

  if (accuracyStat) accuracyStat.textContent = `${(metrics.accuracy * 100).toFixed(1)}%`;
  if (efficiencyStat) efficiencyStat.textContent = `${(metrics.efficiency * 100).toFixed(1)}%`;
  if (complexityStat) complexityStat.textContent = `${metrics.complexity.toFixed(1)}`;
}

function showLoading(elementId: string, message: string) {
  const element = document.getElementById(elementId);
  if (element) {
    element.innerHTML = `
      <div class="loading">
        <div class="spinner"></div>
        ${message}
      </div>
    `;
  }
}

function showError(message: string) {
  const element = document.getElementById('problem-display');
  if (element) {
    element.innerHTML = `
      <div class="error" style="padding: 1rem; border-radius: 8px; text-align: center;">
        ❌ ${message}
      </div>
    `;
  }
}

function hideElements(elementIds: string[]) {
  elementIds.forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = 'none';
    }
  });
}

function showElements(elementIds: string[]) {
  elementIds.forEach(id => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = 'block';
    }
  });
}

function updateUI() {
  // Any initial UI updates can go here
  console.log('UI updated');
}

// Demo mode functions for when Tauri is not available
function showDemoMode() {
  const sampleProblems = {
    arithmetic: { 
      statement: "Calculate: 15 + 27", 
      answer: "42",
      hint: "Add the numbers step by step: 15 + 27 = 42"
    },
    algebra: { 
      statement: "Solve for x: 2x + 5 = 13", 
      answer: "4",
      hint: "Subtract 5 from both sides, then divide by 2"
    },
    calculus: { 
      statement: "Find the derivative of f(x) = x² + 3x", 
      answer: "2x + 3",
      hint: "Use the power rule: d/dx(x^n) = n·x^(n-1)"
    },
    trigonometry: { 
      statement: "Find sin(π/6)", 
      answer: "1/2",
      hint: "π/6 = 30°, which is a special angle"
    },
    statistics: { 
      statement: "Find the mean of [2, 4, 6, 8, 10]", 
      answer: "6",
      hint: "Mean = sum of all values / number of values"
    },
    discrete_math: { 
      statement: "Calculate 5! (5 factorial)", 
      answer: "120",
      hint: "5! = 5 × 4 × 3 × 2 × 1"
    }
  };

  const problem = sampleProblems[selectedTopic as keyof typeof sampleProblems] || sampleProblems.arithmetic;
  
  currentProblem = {
    id: 'demo-' + Date.now(),
    problem_type: selectedTopic,
    statement: problem.statement,
    hint: problem.hint,
    expected_answer: problem.answer,
    difficulty: selectedDifficulty,
    topic: selectedTopic
  };

  displayProblem(currentProblem);
  showDemoNetwork();
  showElements(['answer-input', 'submit-answer', 'show-hint']);

  const answerInput = document.getElementById('answer-input') as HTMLInputElement;
  answerInput.focus();
}

function showDemoSolution(studentAnswer: string) {
  if (!currentProblem) return;

  const isCorrect = studentAnswer.toLowerCase().trim() === currentProblem.expected_answer.toLowerCase().trim();
  
  const response: SolutionResponse = {
    answer: studentAnswer,
    correct: isCorrect,
    explanation: isCorrect ? 
      "Correct! Well done." : 
      `Not quite right. The correct answer is: ${currentProblem.expected_answer}`,
    steps: [
      "Step 1: Analyze the problem",
      "Step 2: Apply the appropriate mathematical method", 
      "Step 3: Calculate the result",
      `Step 4: The answer is ${currentProblem.expected_answer}`
    ],
    network_data: {} as NetworkVisualization
  };

  displaySolution(response);
  showDemoNetwork();
}

function showDemoNetwork() {
  const demoNetwork: NetworkVisualization = {
    nodes: [
      { id: 'input_1', label: 'Input 1', node_type: 'input', value: 0.8, x: 100, y: 150, color: '#4CAF50' },
      { id: 'input_2', label: 'Input 2', node_type: 'input', value: 0.6, x: 100, y: 220, color: '#4CAF50' },
      { id: 'hidden_1', label: 'Hidden 1', node_type: 'hidden', value: 0.7, x: 300, y: 120, color: '#2196F3' },
      { id: 'hidden_2', label: 'Hidden 2', node_type: 'hidden', value: 0.9, x: 300, y: 200, color: '#2196F3' },
      { id: 'hidden_3', label: 'Hidden 3', node_type: 'hidden', value: 0.4, x: 300, y: 280, color: '#2196F3' },
      { id: 'output_1', label: 'Output', node_type: 'output', value: 0.85, x: 500, y: 200, color: '#FF9800' }
    ],
    edges: [
      { from: 'input_1', to: 'hidden_1', weight: 0.8, color: '#666', width: 2.0 },
      { from: 'input_1', to: 'hidden_2', weight: 0.6, color: '#666', width: 2.0 },
      { from: 'input_2', to: 'hidden_1', weight: -0.4, color: '#666', width: 1.5 },
      { from: 'input_2', to: 'hidden_2', weight: 0.9, color: '#666', width: 2.0 },
      { from: 'input_2', to: 'hidden_3', weight: 0.3, color: '#666', width: 1.5 },
      { from: 'hidden_1', to: 'output_1', weight: 0.7, color: '#999', width: 2.0 },
      { from: 'hidden_2', to: 'output_1', weight: 0.5, color: '#999', width: 1.5 },
      { from: 'hidden_3', to: 'output_1', weight: -0.2, color: '#999', width: 1.0 }
    ],
    metrics: {
      accuracy: 0.87,
      efficiency: 0.82,
      complexity: 2.1,
      nodes_count: 6,
      edges_count: 8
    }
  };

  visualizeNetwork(demoNetwork);
  updatePerformanceStats(demoNetwork.metrics);
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', initializeApp);

// Add some sample math rendering with KaTeX (if available)
declare global {
  interface Window {
    katex?: any;
    __TAURI_INTERNALS__?: any;
  }
}

function renderMath(element: HTMLElement) {
  if (window.katex && element.textContent) {
    try {
      const mathContent = element.textContent;
      if (mathContent.includes('\\') || mathContent.includes('^') || mathContent.includes('_')) {
        element.innerHTML = window.katex.renderToString(mathContent, {
          throwOnError: false,
          displayMode: false
        });
      }
    } catch (error) {
      console.warn('Failed to render math:', error);
    }
  }
}

// Export for potential external use
export {
  initializeApp,
  generateProblem,
  submitAnswer,
  visualizeNetwork
};