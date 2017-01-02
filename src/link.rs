use innovation_database::InnovationDatabase;
use network_node::NetworkNode;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::rc::Rc;
use std::sync::RwLock;
use traits::Traits;
use weight::Weight;


#[derive(Clone)]
pub struct Link {
    onode: RefCell<NetworkNode>,
    inode: RefCell<NetworkNode>,
    weight: Weight,
    added_weight: f64,
    time_delay: bool,
    link_trait: Option<Traits>,
    trait_id: u32,
    innovation: u64,
    recur: bool,
}

impl Link {
    /// Create a new synapse
    pub fn new(onode: RefCell<NetworkNode>,
               inode: RefCell<NetworkNode>,
               link_trait: Traits,
               weight: Weight,
               innovation: u64,
               recur: bool)
               -> Link {
        Link {
            onode: onode,
            inode: inode,
            weight: weight,
            added_weight: 0f64,
            time_delay: false,
            trait_id: link_trait.trait_id(),
            link_trait: Some(link_trait),
            innovation: innovation,
            recur: recur,
        }
    }
    /// Create a new synapse with no link trait
    pub fn new_without_trait(onode: RefCell<NetworkNode>,
                             inode: RefCell<NetworkNode>,
                             weight: Weight,
                             innovation: u64,
                             recur: bool)
                             -> Link {
        Link {
            onode: onode,
            inode: inode,
            weight: weight,
            added_weight: 0f64,
            time_delay: false,
            link_trait: None,
            trait_id: 1,
            innovation: innovation,
            recur: recur,
        }
    }

    /// Set link trait
    pub fn set_trait(&mut self, traits: Traits) {
        self.link_trait = Some(traits);
    }

    /// Getter
    pub fn innovation(&self) -> u64 {
        self.innovation
    }

    /// Getter
    pub fn looped_recurrent(&self) -> bool {
        self.onode.borrow().innovation() == self.inode.borrow().innovation()
    }

    /// Getter
    pub fn weight(&self) -> Weight {
        self.weight
    }
    /// Getter
    pub fn recur(&self) -> bool {
        self.recur
    }
    /// Getter
    pub fn inode(&mut self) -> RefCell<NetworkNode> {
        self.inode.clone()
    }
    /// Setter
    pub fn set_added_weight(&mut self, w: f64) {
        self.added_weight = w;
    }
}

impl Eq for Link {}

impl PartialEq for Link {
    fn eq(&self, other: &Link) -> bool {
        self.innovation == other.innovation
    }
}


impl PartialOrd for Link {
    fn partial_cmp(&self, other: &Link) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Link {
    fn cmp(&self, other: &Link) -> Ordering {
        self.innovation.cmp(&other.innovation)
    }
}

mod tests {
    use super::*;
    use weight::Weight;

    #[test]
    fn test_new() {
        let lg1 = Link::new(0, Weight::new(), 0, 0.0, true, true);
        let lg2 = Link::new(0, Weight::new(), 1, 0.0, true, true);
        assert!(lg1.enabled());
        assert!(lg2.enabled());
        assert!(lg1.looped_recurrent());
        assert!(lg2.looped_recurrent());
    }

    #[test]
    fn test_traits() {
        let lg1 = Link::new(0, Weight::new(), 0, 0.0, true, true);
        let lg2 = Link::new(0, Weight::new(), 1, 0.0, true, true);
        assert!(lg1 < lg2);
        assert!(lg2 > lg1);
        assert!(lg2 != lg1);

    }

}
