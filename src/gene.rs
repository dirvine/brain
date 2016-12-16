use link::Link;
use network_node::NetworkNode;
use std::rc::Rc;
use std::sync::RwLock;
use traits::Traits;
use weight::Weight;

/// A gene
#[derive(PartialEq, Clone)]
pub struct Gene {
    lnk: Rc<Link>,
    innovation_num: u64,
    mutation_amount: f64, // Used to see how much mutation has changed the link
    enabled: bool, // When this is off the Gene is disabled
    frozen: bool, // When frozen, the linkweight cannot be mutated
}

impl Gene {
    /// Construct a gene
    pub fn new_no_trait(onode: Rc<NetworkNode>,
                        inode: Rc<NetworkNode>,
                        link_trait: Traits,
                        link_weight: Weight,
                        innov: u64,
                        mnum: f64,
                        recur: bool)
                        -> Gene {
        Gene {
            lnk: Rc::new(Link::new(onode, inode, link_trait, link_weight, innov, recur)),
            innovation_num: innov,
            mutation_amount: mnum,
            enabled: true,
            frozen: false,
        }
    }

    /// Construct a gene off of another gene as a duplicate
    pub fn new_use_existing_gene(onode: Rc<NetworkNode>,
                                 inode: Rc<NetworkNode>,
                                 gene: &Gene,
                                 link_trait: Traits,
                                 recur: bool)
                                 -> Gene {
        Gene {
            lnk: Rc::new(Link::new(onode,
                                   inode,
                                   link_trait,
                                   gene.lnk.weight(),
                                   gene.innovation_num(),
                                   recur)),
            innovation_num: gene.innovation_num(),
            mutation_amount: gene.mutation_amount(),
            enabled: gene.enabled(),
            frozen: gene.frozen(),
        }
    }

    /// Getter
    pub fn innovation_num(&self) -> u64 {
        self.innovation_num
    }
    /// Getter
    pub fn mutation_amount(&self) -> f64 {
        self.mutation_amount
    }
    /// Getter
    pub fn enabled(&self) -> bool {
        self.enabled
    }
    /// Getter
    pub fn frozen(&self) -> bool {
        self.frozen
    }

    // 	//Print gene to a file- called from Genome
    //     void print_to_file(std::ostream &outFile);
    // void print_to_file(std::ofstream &outFile);
}
