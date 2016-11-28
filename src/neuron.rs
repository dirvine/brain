
pub enum Node {
    In,
    Out,
    Hidden,
    Bias,
}

/// Nodes have synapses, these are weighted connections
/// to each other. In Nodes only have out synapses and likewise Out nodes only
/// have In synapses
/// Hidden Nodes have both in and out synapses.
pub struct Neuron {
    node_type: Node,
    out_syn: Vec<f64>,
    in_syn: Vec<f64>,
}

impl Neuron {}

mod tests {}
