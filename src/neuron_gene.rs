use std::time::Duration;

/// All possible neuron types
pub enum NeuronGeneType {
    In,
    Out,
    Hidden,
    Bias,
    NotUsed,
}

/// All possible activation function types
enum ActivationFunction {
    SignedSigmoid, // Sigmoid function   (default) (blurred cutting plane)
    UnSignedSigmoid,
    Tanh,
    TanhCubic,
    SignedStep, // Treshold (0 or 1)  (cutting plane)
    UnSignedStep,
    SignedGauss, // Gaussian           (symmetry)
    UnSignedGauss,
    Abs, // Absolute value |x| (another symettry)
    SignedSine, // Sine wave          (smooth repetition)
    UnSignedSine,
    Linear, // Linear f(x)=x      (combining coordinate frames only)
    Relu, // Rectifiers
    Softplus,
}



/// Nodes have link genes, these are weighted connections
/// to each other. In Nodes only have out link genes and likewise Out nodes only
/// have In link genes
/// Hidden Nodes have both in and out link genes.
pub struct NeuronGene {
    id: u64,
    node_type: NeuronGeneType,
    out_syn: Vec<f64>,
    in_syn: Vec<f64>,
    x: u32, // for plots
    y: u32, // for plots
    net_depth: u32,
    // Additional parameters associated with the
    // neuron's activation function.
    // The current activation function may not use
    // any of them anyway.
    // A is usually used to alter the function's slope with a scalar
    // B is usually used to force a bias to the neuron
    // -------------------
    // Sigmoid : using A, B (slope, shift)
    // Step    : using B    (shift)
    // Gauss   : using A, B (slope, shift))
    // Abs     : using B    (shift)
    // Sine    : using A    (frequency, phase)
    // Square  : using A, B (high phase lenght, low phase length)
    // Linear  : using B    (shift)
    A: u32,
    B: u32,
    // Time constant value used when
    // the neuron is activating in leaky integrator mode
    time_constant: Duration,
    // Bias value used when the neuron is activating in
    // leaky integrator mode
    Bias: f64,
    // The type of activation function the neuron has
    activation_function: ActivationFunction,
}

impl NeuronGene {}

mod tests {}
