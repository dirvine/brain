use neuron_gene::NeuronGeneType;

enum InnovationType {
    Neuron,
    Link,
}

pub struct Innovation {
    id: u32,
    innovation_type: InnovationType,
    from_neuron: u32,
    to_neuron: u32,
    neuron_id: u32,
    neuron_type: NeuronGeneType,
}
