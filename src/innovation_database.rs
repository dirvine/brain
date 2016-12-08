use innovation::{Innovation, InnovationType};

pub struct InnovationDatabase {
    innovation_number: u64,
    innovations: Vec<Innovation>,
}

// impl InnovationDatabase {
//     /// Create new empty db
//     pub fn new() -> InnovationDatabase {
//         InnovationDatabase {
//             innovation_number: 1,
//             innovations: Vec::new(),
//         }
//     }
//     /// Start with initial conditions
// pub fn with_start_Conditions(innovation_number: u64) ->
// InnovationDatabase {
//         InnovationDatabase {
//             innovation_number: innovation_number,
//             innovations: Vec::new(),
//         }
//     }
//
//     /// Has innovation occured already
//     pub fn find_innovation(&self,
//                            from_neuron: u32,
//                            to_neuron: u32,
//                            innovation_type: InnovationType)
//                            -> Option<&Innovation> {
//         self.innovations
//             .iter()
// .find(|x| x.from_neuron() == from_neuron && x.to_neuron() ==
// to_neuron)
//     }
//
//     /// Get last innovation
//     pub fn last_innovation(&self) -> Option<&Innovation> {
//         self.innovations.last()
//     }
//
//     /// Add `NeuronGene`
//     /// If there was an existing link this is split
//     pub fn add_neuron_innovation(&mut self,
//                                  from_n: u32,
//                                  to_n: u32,
//                                  innovation_type: InnovationType)
//                                  -> u32 {
//         self.innovations
// .push(Innovation::new(from_n, to_n, self.next_neuron_id,
// innovation_type));
//         let neuron_id = self.next_neuron_id;
//         self.next_neuron_id += 1;
//         self.next_innovation_number += 1;
//         neuron_id;
//     }
//
//     /// Add `LingGene`
// pub fn add_link_innovation(&mut self, from_neuron: u32, to_neuron: u32)
// {}
// }
