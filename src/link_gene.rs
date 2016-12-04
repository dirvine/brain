use neuron_gene::NeuronGene;
use std::cmp::Ordering;

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
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Getter
    pub fn recurrent(&self) -> bool {
        self.recurrent
    }

    /// Getter
    pub fn looped_recurrent(&self) -> bool {
        self.from_neuron == self.to_neuron
    }


    /// Getter
    pub fn innovation(&self) -> u64 {
        self.innovation
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

impl Eq for LinkGene {}

impl PartialEq for LinkGene {
    fn eq(&self, other: &LinkGene) -> bool {
        self.innovation == other.innovation
    }
}


impl PartialOrd for LinkGene {
    fn partial_cmp(&self, other: &LinkGene) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LinkGene {
    fn cmp(&self, other: &LinkGene) -> Ordering {
        self.innovation.cmp(&other.innovation)
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let lg1 = LinkGene::new(0, 0, 0, 0.0, true, true);
        let lg2 = LinkGene::new(0, 0, 1, 0.0, true, true);
        assert!(lg1.enabled());
        assert!(lg2.enabled());
        assert!(lg1.looped_recurrent());
        assert!(lg2.looped_recurrent());
        assert!(lg1.recurrent());
        assert!(lg2.recurrent());
    }

    #[test]
    fn test_traits() {
        let lg1 = LinkGene::new(0, 0, 0, 0.0, true, true);
        let lg2 = LinkGene::new(0, 0, 1, 0.0, true, true);
        assert!(lg1 < lg2);
        assert!(lg2 > lg1);
        assert!(lg2 != lg1);

    }

}
