use link::Link;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use traits::Traits;

/// Node types
#[derive(PartialEq, PartialOrd, Clone)]
pub enum NodeType {
    Sensor,
    Input,
    Output,
    Bias,
}

/// Activation function types
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
enum ActivationFunction {
    SignedSigmoid, // Sigmoid function   (default) (blurred cutting plane)
    UnSignedSigmoid,
    Tanh,
    TanhCubic,
    SignedStep, // Treshold (0 or 1)  (cutting plane)
    UnSignedStep,
    SignedGauss, // Gaussian           (symmetry)
    UnSignedGauss,
    Abs, // Absolute value |x| (another symettry)
    SignedSine, // Sine wave          (smooth repetition)
    UnSignedSine,
    Linear, // Linear f(x)=x      (combining coordinate frames only)
    Relu, // Rectifiers
    Softplus,
}


/// Nodes have links these are weighted connections
/// to each other. In Nodes only have out links and likewise Out nodes only
/// have In links
/// Hidden Nodes have both in and out links.
/// Use an activation count to avoid flushing
#[derive(Clone)]
pub struct NetworkNode {
    innovation: u64,
    activation_count: u32, // keeps track of which activation the node is currently in
    last_activation: f64, // Holds the previous step's activation for recurrency
    penultimate_activation: f64, // Holds the activation BEFORE the prevous step's
    // This is necessary for a special recurrent case when the innode
    // of a recurrent link is one time step ahead of the outnode.
    // The innode then needs to send from TWO time steps ago
    nodetrait: Traits, // Points to a trait of parameters
    trait_id: u32, // identify the trait derived by this node
    dup: Arc<Mutex<NetworkNode>>, // Used for Genome duplication
    analogue: Arc<Mutex<NetworkNode>>, // Used for Gene decoding
    overriden: bool, // The NNode cannot compute its own output- something is overriding it
    override_value: f64, // Contains the activation value that will override this node's activation
    // Pointer to the Sensor corresponding to this Body.
    // Sensor* mySensor;
    frozen: bool, // When frozen, cannot be mutated (meaning its trait pointer is fixed)
    activation_function: ActivationFunction, // type is either SIGMOID ..or others that can be added
    nodetype: NodeType,
    activesum: f64, // The incoming activity before being processed
    activation: f64, // The total activation entering the NNode
    active_flag: bool, // To make sure outputs are active
    // ************ LEARNING PARAMETERS ***********
    // The following parameters are for use in
    //   neurons that learn through habituation,
    //   sensitization, or Hebbian-type processes
    params: Vec<f64>,
    incoming: Vec<RefCell<Link>>, /* A list of pointers to incoming weighted signals from other
                                   * nodes */
    outgoing: Vec<RefCell<Link>>, // A list of pointers to links carrying this node's signal
}

impl NetworkNode {
    /// Getter
    pub fn node_type(&self) -> &NodeType {
        &self.nodetype
    }
    /// Set link trait
    pub fn set_node_trait(&mut self, traits: Traits) {
        self.nodetrait = traits;
    }
    /// how many outgoing links has this node
    pub fn num_outgoing_links(&self) -> usize {
        self.outgoing.len()
    }
    /// how many incoming links has this node
    pub fn num_incoming_links(&self) -> usize {
        self.incoming.len()
    }
    /// How many links in total
    pub fn total_links(&self) -> usize {
        self.incoming.len() + self.outgoing.len()
    }

    /// Allows alteration between NEURON and SENSOR.
    pub fn set_type(&mut self, nodetype: NodeType) {
        self.nodetype = nodetype;
    }

    /// Getter
    pub fn innovation(&self) -> u64 {
        self.innovation
    }
    /// Getter
    pub fn frozen(&self) -> bool {
        self.frozen
    }

    /// Reset activations
    pub fn flush(&mut self) {
        self.activation_count = 0;
        self.last_activation = 0f64;
        self.penultimate_activation = 0f64;
        self.activation = 0f64;
    }
    /// If the node is a SENSOR, returns true and loads the value
    pub fn sensor_load(&mut self, load: f64) -> bool {
        match self.nodetype {
            NodeType::Sensor => {
                self.penultimate_activation = self.last_activation;
                self.last_activation = self.activation;
                self.activation_count += 1;
                self.activation = load;
                true
            }
            _ => false,
        }
    }
    /// Note: NEAT keeps track of which links are recurrent and which
    /// are not even though this is unnecessary for activation.
    /// It is useful to do so for 2 other reasons:
    /// 1. It makes networks visualization of recurrent networks possible
    /// 2. It allows genetic control of the proportion of connections
    ///    that may become recurrent
    /// Add an incoming connection a node
    pub fn add_incoming_connection(&mut self, node: RefCell<Link>) {
        self.incoming.push(node);
    }
    /// Add an outgoing connection
    pub fn add_outgoing_connection(&mut self, node: RefCell<Link>) {
        self.outgoing.push(node);
    }
    /// Add a recurrent connection
    pub fn add_recurrent_connection(&mut self, node: RefCell<Link>) {
        self.outgoing.push(node.clone());
        self.incoming.push(node.clone());
    }
    /// Getter
    pub fn activation(&self) -> f64 {
        self.activation
    }
    /// Getter
    pub fn last_Activation(&self) -> f64 {
        self.last_activation
    }
    /// Getter
    pub fn penultimate_activation(&self) -> f64 {
        self.penultimate_activation
    }
    /// This recursively flushes everything leading into and including this NNode, including recurrencies
    pub fn flush_back(&mut self) {
        if self.activation_count > 0 {
            self.flush();
        }
        if *self.node_type() == NodeType::Sensor {
            self.flush();
            return;
        }
        self.incoming.iter_mut().map(|ref mut link| {
            link.borrow_mut().set_added_weight(0f64);
            // TODO link.borrow().inode().borrow_mut().flush_back();
        });
    }
    /// The Gene that created this node
    pub fn analogue(&self) -> Arc<Mutex<NetworkNode>> {
        self.analogue.clone()
    }
    /// Override this nodes output
    pub fn override_output(&mut self, new_output: f64) {
        self.override_value = new_output;
        self.overriden = true;
    }
    /// Is this node overridden
    pub fn overridden(&self) -> bool {
        self.overriden
    }
    /// activate the override
    pub fn activate_override(&mut self) {
        self.activation = self.override_value;
        self.overriden = false;
    }
}

impl Eq for NetworkNode {}

impl PartialEq for NetworkNode {
    fn eq(&self, other: &NetworkNode) -> bool {
        self.innovation == other.innovation
    }
}


impl PartialOrd for NetworkNode {
    fn partial_cmp(&self, other: &NetworkNode) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NetworkNode {
    fn cmp(&self, other: &NetworkNode) -> Ordering {
        self.innovation.cmp(&other.innovation)
    }
}
mod tests {}
