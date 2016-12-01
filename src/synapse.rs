use Neuron:

struct Synapse {
from: u32,
to: u32,
innovation: u64,
weight: f64,
recurrent: bool,
    enabled: bool
}

impl Synapse {
    /// Create a new synapse
    pub fn new(from: u32, to: u32, innovation: u64, weight: f64, recurrent: bool, enabled: bool) -> Synapse {
        Synapse {
            from: from,
            to: to,
            innovation: innovation,
            weight: weight,
            recurrent: recurrent,
            enabled: enabled
        }
    }
    ///Getter
    pub from(&self) -> u32 {
    self.from}
    pub to(&self) -> u32 {
    self.to}

}
