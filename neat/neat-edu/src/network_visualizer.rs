use crate::{NetworkVisualization, NetworkNode, NetworkEdge, NetworkMetrics};

pub fn create_network_visualization(topic: &str) -> NetworkVisualization {
    let node_count = match topic {
        "arithmetic" => 5,
        "algebra" => 7,
        "calculus" => 10,
        "trigonometry" => 8,
        "statistics" => 6,
        _ => 6,
    };
    
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    
    // Create input layer
    for i in 0..3 {
        nodes.push(NetworkNode {
            id: format!("input_{}", i),
            label: format!("Input {}", i + 1),
            node_type: "input".to_string(),
            value: 0.5 + (i as f64 * 0.1),
            x: 50.0,
            y: 100.0 + (i as f64 * 100.0),
            color: "#4CAF50".to_string(),
        });
    }
    
    // Create hidden layer
    for i in 0..node_count-5 {
        nodes.push(NetworkNode {
            id: format!("hidden_{}", i),
            label: format!("Hidden {}", i + 1),
            node_type: "hidden".to_string(),
            value: 0.3 + (i as f64 * 0.15),
            x: 250.0,
            y: 150.0 + (i as f64 * 80.0),
            color: "#2196F3".to_string(),
        });
        
        // Connect to inputs
        for j in 0..3 {
            edges.push(NetworkEdge {
                from: format!("input_{}", j),
                to: format!("hidden_{}", i),
                weight: 0.2 + (i as f64 * 0.1),
                color: "#666".to_string(),
                width: 2.0,
            });
        }
    }
    
    // Create output layer
    for i in 0..2 {
        nodes.push(NetworkNode {
            id: format!("output_{}", i),
            label: format!("Output {}", i + 1),
            node_type: "output".to_string(),
            value: 0.8 + (i as f64 * 0.1),
            x: 450.0,
            y: 175.0 + (i as f64 * 100.0),
            color: "#FF9800".to_string(),
        });
        
        // Connect to hidden nodes
        for j in 0..node_count-5 {
            edges.push(NetworkEdge {
                from: format!("hidden_{}", j),
                to: format!("output_{}", i),
                weight: 0.4 + (j as f64 * 0.05),
                color: "#999".to_string(),
                width: 1.5,
            });
        }
    }
    
    let edges_count = edges.len();
    NetworkVisualization {
        nodes,
        edges,
        metrics: NetworkMetrics {
            accuracy: match topic {
                "arithmetic" => 0.94,
                "algebra" => 0.89,
                "calculus" => 0.92,
                "trigonometry" => 0.87,
                "statistics" => 0.91,
                "discrete math" => 0.88,
                _ => 0.90,
            },
            efficiency: match topic {
                "arithmetic" => 0.87,
                "algebra" => 0.82,
                "calculus" => 0.85,
                "trigonometry" => 0.79,
                "statistics" => 0.84,
                "discrete math" => 0.81,
                _ => 0.83,
            },
            complexity: node_count as f64 * 0.1,
            nodes_count: node_count,
            edges_count,
        },
    }
}