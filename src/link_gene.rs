use neuron_gene::NeuronGene;

const MIN_WEIGHT: f64 = -1f64;
const MAX_WEIGHT: f64 = 1f64;

pub struct LinkGene {
    from_neuron: u32,
    to_neuron: u32,
    innovation: u64,
    weight: f64,
    recurrent: bool,
    enabled: bool,
}

impl LinkGene {
    /// Create a new synapse
    pub fn new(from_neuron: u32,
               to_neuron: u32,
               innovation: u64,
               weight: f64,
               recurrent: bool,
               enabled: bool)
               -> LinkGene {
        assert!(MIN_WEIGHT <= weight && weight <= MAX_WEIGHT);
        LinkGene {
            from_neuron: from_neuron,
            to_neuron: to_neuron,
            innovation: innovation,
            weight: weight,
            recurrent: recurrent,
            enabled: enabled,
        }
    }
    /// Getter
    pub fn from_neuron(&self) -> u32 {
        self.from_neuron
    }

    /// Getter
    pub fn to_neuron(&self) -> u32 {
        self.to_neuron
    }

    /// Getter
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Setter
    pub fn set_weight(&mut self, weight: f64) {
        assert!(MIN_WEIGHT <= weight && weight <= MAX_WEIGHT);
        self.weight = weight;
    }
}
