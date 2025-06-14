// Embedded frontend for the NEAT Educational Platform

pub const HTML_CONTENT: &str = r#"
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>NEAT Educational Platform</title>
    <style>
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
        min-height: 100vh;
        display: flex;
        padding: 20px;
        gap: 20px;
      }

      .panel {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
      }

      .left-panel {
        flex: 1;
        max-width: 600px;
      }

      .right-panel {
        flex: 1;
        min-width: 600px;
      }

      h1, h2 {
        color: #4a5568;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }

      .topic-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
      }

      .topic-btn {
        padding: 0.5rem 1rem;
        border: 2px solid #e2e8f0;
        border-radius: 25px;
        background: white;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
        font-weight: 500;
      }

      .topic-btn:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
      }

      .topic-btn.active {
        background: #667eea;
        color: white;
        border-color: #667eea;
      }

      .difficulty-section {
        margin: 1.5rem 0;
      }

      .difficulty-slider {
        width: 100%;
        height: 8px;
        border-radius: 5px;
        background: #e2e8f0;
        outline: none;
        margin: 1rem 0;
      }

      .difficulty-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #718096;
      }

      .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1.5rem 0;
      }

      .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
      }

      .problem-display {
        background: #f7fafc;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        min-height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #718096;
        font-style: italic;
      }

      .answer-section {
        margin-top: 1rem;
        display: none;
      }

      .answer-section.show {
        display: block;
      }

      .answer-input {
        width: 100%;
        padding: 0.75rem;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        font-size: 1rem;
        margin: 0.5rem 0;
        transition: border-color 0.3s ease;
      }

      .answer-input:focus {
        outline: none;
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
      }

      .submit-btn {
        background: #48bb78;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-right: 0.5rem;
      }

      .submit-btn:hover {
        background: #38a169;
        transform: translateY(-1px);
      }

      .hint-btn {
        background: #ed8936;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
      }

      .hint-btn:hover {
        background: #dd6b20;
        transform: translateY(-1px);
      }

      .metrics {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
      }

      .metric {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
      }

      .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
      }

      .metric-label {
        font-size: 0.8rem;
        opacity: 0.9;
      }

      .network-container {
        height: 400px;
        background: white;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #718096;
        font-style: italic;
      }

      .explanation {
        background: #f0fff4;
        border: 1px solid #9ae6b4;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        display: none;
      }

      .explanation.show {
        display: block;
      }

      .error {
        background: #fed7d7;
        border: 1px solid #fc8181;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #c53030;
      }
    </style>
  </head>
  <body>
    <div class="panel left-panel">
      <h1>🎯 Mathematical Problem Solver</h1>
      
      <div class="topic-buttons">
        <button class="topic-btn active" data-topic="arithmetic">Arithmetic</button>
        <button class="topic-btn" data-topic="algebra">Algebra</button>
        <button class="topic-btn" data-topic="calculus">Calculus</button>
        <button class="topic-btn" data-topic="trigonometry">Trigonometry</button>
        <button class="topic-btn" data-topic="statistics">Statistics</button>
        <button class="topic-btn" data-topic="discrete math">Discrete Math</button>
      </div>

      <div class="difficulty-section">
        <label><strong>Difficulty Level:</strong></label>
        <input type="range" class="difficulty-slider" id="difficulty" min="0" max="2" value="0" step="1">
        <div class="difficulty-labels">
          <span>Easy</span>
          <span>Medium</span>
          <span>Hard</span>
        </div>
      </div>

      <button class="generate-btn" id="generate">🧠 Generate New Problem</button>

      <div class="problem-display" id="problem">
        Generate a problem to see the neural network in action!
      </div>

      <div class="answer-section" id="answer-section">
        <input type="text" class="answer-input" id="answer" placeholder="Enter your answer here...">
        <div>
          <button class="submit-btn" id="submit">✓ Submit Answer</button>
          <button class="hint-btn" id="hint">💡 Show Hint</button>
        </div>
      </div>

      <div class="explanation" id="explanation"></div>
    </div>

    <div class="panel right-panel">
      <h2>🧠 Neural Network Visualization</h2>
      
      <div class="metrics">
        <div class="metric">
          <div class="metric-value" id="accuracy">--</div>
          <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric">
          <div class="metric-value" id="efficiency">--</div>
          <div class="metric-label">Efficiency</div>
        </div>
        <div class="metric">
          <div class="metric-value" id="complexity">--</div>
          <div class="metric-label">Complexity</div>
        </div>
      </div>

      <div class="network-container" id="network">
        Generate a problem to see the neural network in action!
      </div>

      <div style="margin-top: 1rem; padding: 1rem; background: #f7fafc; border-radius: 10px;">
        <strong>🧠 How It Works:</strong><br>
        Each mathematical problem is solved by a specialized neural network that has evolved 
        using NEAT (NeuroEvolution of Augmenting Topologies). The network structure grows and 
        adapts to handle different types of mathematical reasoning. Watch how the network 
        architecture changes based on problem complexity!
      </div>
    </div>

    <script>
      // Simple demo functionality
      let selectedTopic = 'arithmetic';
      let selectedDifficulty = 'easy';
      let currentProblem = null;

      // Topic selection
      document.querySelectorAll('.topic-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('.topic-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          selectedTopic = btn.dataset.topic;
        });
      });

      // Difficulty selection
      document.getElementById('difficulty').addEventListener('change', (e) => {
        const levels = ['easy', 'medium', 'hard'];
        selectedDifficulty = levels[e.target.value];
      });

      // Generate problem
      document.getElementById('generate').addEventListener('click', async () => {
        try {
          console.log('Generating problem for:', selectedTopic, selectedDifficulty);
          
          // Call Tauri backend
          const response = await window.__TAURI__.invoke('generate_problem', {
            request: {
              topic: selectedTopic,
              difficulty: selectedDifficulty,
              problem_type: 'standard'
            }
          });

          currentProblem = response;
          displayProblem(response);
          updateMetrics(response.network_data.metrics);
          showAnswerSection();
          
        } catch (error) {
          console.error('Error generating problem:', error);
          showError('Failed to generate problem: ' + error);
        }
      });

      // Submit answer
      document.getElementById('submit').addEventListener('click', async () => {
        const answer = document.getElementById('answer').value.trim();
        if (!answer || !currentProblem) return;

        try {
          const response = await window.__TAURI__.invoke('solve_problem', {
            request: {
              problem: currentProblem.problem_text,
              student_answer: answer
            }
          });

          showExplanation(response);
          
        } catch (error) {
          console.error('Error submitting answer:', error);
          showError('Failed to submit answer: ' + error);
        }
      });

      function displayProblem(problem) {
        const problemDiv = document.getElementById('problem');
        problemDiv.innerHTML = '<div style="font-size: 1.2rem; font-weight: bold;">' + problem.problem_text + '</div>';
        problemDiv.style.fontStyle = 'normal';
        problemDiv.style.color = '#2d3748';
      }

      function updateMetrics(metrics) {
        document.getElementById('accuracy').textContent = (metrics.accuracy * 100).toFixed(1) + '%';
        document.getElementById('efficiency').textContent = (metrics.efficiency * 100).toFixed(1) + '%';
        document.getElementById('complexity').textContent = metrics.complexity.toFixed(1);
      }

      function showAnswerSection() {
        document.getElementById('answer-section').classList.add('show');
        document.getElementById('answer').value = '';
        document.getElementById('answer').focus();
      }

      function showExplanation(response) {
        const explanationDiv = document.getElementById('explanation');
        const icon = response.correct ? '✅' : '❌';
        const status = response.correct ? 'Correct!' : 'Incorrect';
        
        explanationDiv.innerHTML = `
          <div style="font-weight: bold; margin-bottom: 0.5rem;">
            ${icon} ${status}
          </div>
          <div>${response.explanation}</div>
        `;
        explanationDiv.classList.add('show');
      }

      function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        document.querySelector('.left-panel').appendChild(errorDiv);
        
        setTimeout(() => errorDiv.remove(), 5000);
      }

      // Initialize
      console.log('NEAT Educational Platform loaded');
    </script>
  </body>
</html>
"#;