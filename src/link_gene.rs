use innovation_database::InnovationDatabase;
use neuron_gene::NeuronGene;
use std::cmp::Ordering;
use std::sync::{Arc, Mutex};
use traits::Traits;

const MIN_WEIGHT: f64 = -1f64;
const MAX_WEIGHT: f64 = 1f64;

pub struct LinkGene {
    from_neuron: Arc<Mutex<NeuronGene>>,
    to_neuron: Arc<Mutex<NeuronGene>>,
    weight: f64,
    added_weight: f64,
    recurrent: bool,
    time_delay: bool,
    link_trait: Arc<Mutex<Traits>>,
    innovation: u64,
}

impl LinkGene {
    /// Create a new synapse
    pub fn new(link_trait: Arc<Mutex<Traits>>,
               from_neuron: Arc<Mutex<NeuronGene>>,
               to_neuron: Arc<Mutex<NeuronGene>>,
               weight: f64,
               recurrent: bool,
               innovation: u64)
               -> LinkGene {
        assert!(MIN_WEIGHT <= weight && weight <= MAX_WEIGHT);
        LinkGene {
            from_neuron: from_neuron.clone(),
            to_neuron: to_neuron.clone(),
            weight: weight,
            added_weight: 0f64,
            recurrent: recurrent,
            time_delay: false,
            link_trait: link_trait,
            innovation: innovation,
        }
    }
    /// Getter
    pub fn from_neuron(&self) -> Arc<Mutex<NeuronGene>> {
        self.from_neuron.clone()
    }

    /// Getter
    pub fn to_neuron(&self) -> Arc<Mutex<NeuronGene>> {
        self.to_neuron.clone()
    }

    /// Getter
    pub fn recurrent(&self) -> bool {
        self.recurrent
    }

    /// Getter
    pub fn innovation(&self) -> u64 {
        self.innovation
    }

    /// Getter
    pub fn looped_recurrent(&self) -> bool {
        self.from_neuron.lock().unwrap().innovation() == self.to_neuron.lock().unwrap().innovation()
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
