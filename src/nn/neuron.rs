use neuron_gene::NeuronGeneType;
use rulinalg::matrix::Matrix;

#[define(Debug, PartialEq)]
struct Neuron {
    double m_activesum : f64,  // the synaptic input
    double m_activation : f64, // the synaptic input passed through the activation function

    double m_a, m_b, m_timeconst, m_bias : f64, // misc parameters
    double m_membrane_potential : f64, // used in leaky integrator mode
    activation_function: ActivationFunction,

    // display variables
    x: f64,
    y: f64,
    z : f64,
    sx : f64,
    sy : f64,
    sz : f64,
    substrate_coords : Vec<f64>
    double m_split_y : f64,
    neuron_type : NeuronGeneType,

    // the sensitivity matrix of this neuron (for RTRL learning)
    sensitivity_matrix: Vec<Vec<f64>>,

}
