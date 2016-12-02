use innovation::Innovation;

struct InnovationDatabase {
    next_neuron_id: u32,
    next_innovaiton_number: u64,
    innovations: Vec<Innovation>,
}
