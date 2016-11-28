use rulinalg::vector::Vector;
use neuron::Neuron;

enum CortexType {
    Nervous,
    Visual,
    Speech,
    Hearing
}

pub Struct Cortex {
    cortex_type: CortexType,
    neurons : Vector<Neuron>,
}

impl Cortex {


}
