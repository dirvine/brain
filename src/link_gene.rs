use neuron_gene::NeuronGene;

pub struct LinkGene {
    from: u32,
    to: u32,
    innovation: u64,
    weight: f64,
    recurrent: bool,
    enabled: bool,
}

impl LinkGene {
    /// Create a new synapse
    pub fn new(from: u32,
               to: u32,
               innovation: u64,
               weight: f64,
               recurrent: bool,
               enabled: bool)
               -> LinkGene {
        LinkGene {
            from: from,
            to: to,
            innovation: innovation,
            weight: weight,
            recurrent: recurrent,
            enabled: enabled,
        }
    }
    /// Getter
    pub fn from(&self) -> u32 {
        self.from
    }
    pub fn to(&self) -> u32 {
        self.to
    }
}
