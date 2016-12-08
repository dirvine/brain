	/// A NETWORK is a LIST of input NODEs and a LIST of output NODEs
	///   The point of the network is to define a single entity which can evolve
	///   or learn on its own, even though it may be part of a larger framework
struct Network {
		int numnodes; // The number of nodes in the net (-1 means not yet counted)
		int numlinks; //The number of links in the net (-1 means not yet counted)

		std::vector<NNode*> all_nodes;  // A list of all the nodes

		std::vector<NNode*>::iterator input_iter;  // For GUILE network inputting

		void destroy();  // Kills all nodes and links within
		void destroy_helper(NNode *curnode,std::vector<NNode*> &seenlist); // helper for above

		void nodecounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist);
		void linkcounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist);

	public:

		Genome *genotype;  // Allows Network to be matched with its Genome

		char *name; // Every Network or subNetwork can have a name
		std::vector<NNode*> inputs;  // NNodes that input into the network
		std::vector<NNode*> outputs; // Values output by the network

		int net_id; // Allow for a network id

		double maxweight; // Maximum weight in network for adaptation purposes

		bool adaptable; // Tells whether network can adapt or not

		// This constructor allows the input and output lists to be supplied
		// Defaults to not using adaptation
		Network(std::vector<NNode*> in,std::vector<NNode*> out,std::vector<NNode*> all,int netid);

		//Same as previous constructor except the adaptibility can be set true or false with adaptval
		Network(std::vector<NNode*> in,std::vector<NNode*> out,std::vector<NNode*> all,int netid, bool adaptval);

		// This constructs a net with empty input and output lists
		Network(int netid);

		//Same as previous constructor except the adaptibility can be set true or false with adaptval
		Network(int netid, bool adaptval);

		// Copy Constructor
		Network(const Network& network);

		~Network();

		// Puts the network back into an inactive state
		void flush();

		// Verify flushedness for debugging
		void flush_check();

		// Activates the net such that all outputs are active
		bool activate();

		// Prints the values of its outputs
		void show_activation();

		void show_input();

		// Add a new input node
		void add_input(NNode*);

		// Add a new output node
		void add_output(NNode*);

		// Takes an array of sensor values and loads it into SENSOR inputs ONLY
		void load_sensors(double*);
		void load_sensors(const std::vector<float> &sensvals);

		// Takes and array of output activations and OVERRIDES the outputs' actual
		// activations with these values (for adaptation)
		void override_outputs(double*);

		// Name the network
		void give_name(char*);

		// Counts the number of nodes in the net if not yet counted
		int nodecount();

		// Counts the number of links in the net if not yet counted
		int linkcount();

		// This checks a POTENTIAL link between a potential in_node
		// and potential out_node to see if it must be recurrent
		// Use count and thresh to jump out in the case of an infinite loop
		bool is_recur(NNode *potin_node,NNode *potout_node,int &count,int thresh);

		// Some functions to help GUILE input into Networks
		int input_start();
		int load_in(double d);

		// If all output are not active then return true
		bool outputsoff();

		// Just print connections weights with carriage returns
		void print_links_tofile(char *filename);

		int max_depth();

		bool is_rec_helper(NNode* curnode, NNode* find_node,std::vector<NNode*> &seenlist);
		bool is_recur2(NNode *potin_node,NNode *potout_node,int &count, int thresh);
	}


// Network::Network(std::vector<NNode*> in,std::vector<NNode*> out,std::vector<NNode*> all,int netid) {
//   inputs=in;
//   outputs=out;
//   all_nodes=all;
//   name=0;   //Defaults to no name  ..NOTE: TRYING TO PRINT AN EMPTY NAME CAN CAUSE A CRASH
//   numnodes=-1;
//   numlinks=-1;
//   net_id=netid;
//   adaptable=false;
// }
//
// Network::Network(std::vector<NNode*> in,std::vector<NNode*> out,std::vector<NNode*> all,int netid, bool adaptval) {
//   inputs=in;
//   outputs=out;
//   all_nodes=all;
//   name=0;   //Defaults to no name  ..NOTE: TRYING TO PRINT AN EMPTY NAME CAN CAUSE A CRASH
//   numnodes=-1;
//   numlinks=-1;
//   net_id=netid;
//   adaptable=adaptval;
// }
//
//
// Network::Network(int netid) {
// 			name=0; //Defaults to no name
// 			numnodes=-1;
// 			numlinks=-1;
// 			net_id=netid;
// 			adaptable=false;
// 		}
//
// Network::Network(int netid, bool adaptval) {
//   name=0; //Defaults to no name
//   numnodes=-1;
//   numlinks=-1;
//   net_id=netid;
//   adaptable=adaptval;
// }
//
//
// Network::Network(const Network& network)
// {
// 	std::vector<NNode*>::const_iterator curnode;
//
// 	// Copy all the inputs
// 	for(curnode = network.inputs.begin(); curnode != network.inputs.end(); ++curnode) {
// 		NNode* n = new NNode(**curnode);
// 		inputs.push_back(n);
// 		all_nodes.push_back(n);
// 	}
//
// 	// Copy all the outputs
// 	for(curnode = network.outputs.begin(); curnode != network.outputs.end(); ++curnode) {
// 		NNode* n = new NNode(**curnode);
// 		outputs.push_back(n);
// 		all_nodes.push_back(n);
// 	}
//
// 	if(network.name)
// 		name = strdup(network.name);
// 	else
// 		name = 0;
//
// 	numnodes = network.numnodes;
// 	numlinks = network.numlinks;
// 	net_id = network.net_id;
// 	adaptable = network.adaptable;
// }
//
// Network::~Network() {
// 			if (name!=0)
// 				delete [] name;
//
// 			destroy();  // Kill off all the nodes and links
//
// 		}
//
// // Puts the network back into an initial state
// void Network::flush() {
// 	std::vector<NNode*>::iterator curnode;
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 		(*curnode)->flushback();
// 	}
// }
//
// // Debugger: Checks network state
// void Network::flush_check() {
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
//         location= std::find(seenlist.begin(),seenlist.end(),(*curnode));
// 		if (location==seenlist.end()) {
// 			seenlist.push_back(*curnode);
// 			(*curnode)->flushback_check(seenlist);
// 		}
// 	}
// }
//
// // If all output are not active then return true
// bool Network::outputsoff() {
// 	std::vector<NNode*>::iterator curnode;
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 		if (((*curnode)->activation_count)==0) return true;
// 	}
//
// 	return false;
// }
//
// // Print the connections weights to a file separated by only carriage returns
// void Network::print_links_tofile(char *filename) {
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<Link*>::iterator curlink;
//
//     std::ofstream oFile(filename);
//
// 	//Make sure it worked
// 	//if (!oFile) {
// 	//	cerr<<"Can't open "<<filename<<" for output"<<endl;
// 		//return 0;
// 	//}
//
// 	for(curnode=all_nodes.begin();curnode!=all_nodes.end();++curnode) {
// 		if (((*curnode)->type)!=SENSOR) {
// 			for(curlink=((*curnode)->incoming).begin(); curlink!=((*curnode)->incoming).end(); ++curlink) {
//                 oFile << (*curlink)->in_node->node_id << " -> " <<( *curlink)->out_node->node_id << " : " << (*curlink)->weight << std::endl;
// 			} // end for loop on links
// 		} //end if
// 	} //end for loop on nodes
//
// 	oFile.close();
//
// } //print_links_tofile
//
// // Activates the net such that all outputs are active
// // Returns true on success;
// bool Network::activate() {
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<Link*>::iterator curlink;
// 	double add_amount;  //For adding to the activesum
// 	bool onetime; //Make sure we at least activate once
// 	int abortcount=0;  //Used in case the output is somehow truncated from the network
//
// 	//cout<<"Activating network: "<<this->genotype<<endl;
//
// 	//Keep activating until all the outputs have become active
// 	//(This only happens on the first activation, because after that they
// 	// are always active)
//
// 	onetime=false;
//
// 	while(outputsoff()||!onetime) {
//
// 		++abortcount;
//
// 		if (abortcount==20) {
// 			return false;
// 			//cout<<"Inputs disconnected from output!"<<endl;
// 		}
// 		//std::cout<<"Outputs are off"<<std::endl;
//
// 		// For each node, compute the sum of its incoming activation
// 		for(curnode=all_nodes.begin();curnode!=all_nodes.end();++curnode) {
// 			//Ignore SENSORS
//
// 			//cout<<"On node "<<(*curnode)->node_id<<endl;
//
// 			if (((*curnode)->type)!=SENSOR) {
// 				(*curnode)->activesum=0;
// 				(*curnode)->active_flag=false;  //This will tell us if it has any active inputs
//
// 				// For each incoming connection, add the activity from the connection to the activesum
// 				for(curlink=((*curnode)->incoming).begin();curlink!=((*curnode)->incoming).end();++curlink) {
// 					//Handle possible time delays
// 					if (!((*curlink)->time_delay)) {
// 						add_amount=((*curlink)->weight)*(((*curlink)->in_node)->get_active_out());
// 						if ((((*curlink)->in_node)->active_flag)||
// 							(((*curlink)->in_node)->type==SENSOR)) (*curnode)->active_flag=true;
// 						(*curnode)->activesum+=add_amount;
// 						//std::cout<<"Node "<<(*curnode)->node_id<<" adding "<<add_amount<<" from node "<<((*curlink)->in_node)->node_id<<std::endl;
// 					}
// 					else {
// 						//Input over a time delayed connection
// 						add_amount=((*curlink)->weight)*(((*curlink)->in_node)->get_active_out_td());
// 						(*curnode)->activesum+=add_amount;
// 					}
//
// 				} //End for over incoming links
//
// 			} //End if (((*curnode)->type)!=SENSOR)
//
// 		} //End for over all nodes
//
// 		// Now activate all the non-sensor nodes off their incoming activation
// 		for(curnode=all_nodes.begin();curnode!=all_nodes.end();++curnode) {
//
// 			if (((*curnode)->type)!=SENSOR) {
// 				//Only activate if some active input came in
// 				if ((*curnode)->active_flag) {
// 					//cout<<"Activating "<<(*curnode)->node_id<<" with "<<(*curnode)->activesum<<": ";
//
// 					//Keep a memory of activations for potential time delayed connections
// 					(*curnode)->last_activation2=(*curnode)->last_activation;
// 					(*curnode)->last_activation=(*curnode)->activation;
//
// 					//If the node is being overrided from outside,
// 					//stick in the override value
// 					if ((*curnode)->overridden()) {
// 						//Set activation to the override value and turn off override
// 						(*curnode)->activate_override();
// 					}
// 					else {
// 						//Now run the net activation through an activation function
// 						if ((*curnode)->ftype==SIGMOID)
// 							(*curnode)->activation=NEAT::fsigmoid((*curnode)->activesum,4.924273,2.4621365);  //Sigmoidal activation- see comments under fsigmoid
// 					}
// 					//cout<<(*curnode)->activation<<endl;
//
// 					//Increment the activation_count
// 					//First activation cannot be from nothing!!
// 					(*curnode)->activation_count++;
// 				}
// 			}
// 		}
//
// 		onetime=true;
// 	}
//
// 	if (adaptable) {
//
// 	  //std::cout << "ADAPTING" << std:endl;
//
// 	  // ADAPTATION:  Adapt weights based on activations
// 	  for(curnode=all_nodes.begin();curnode!=all_nodes.end();++curnode) {
// 	    //Ignore SENSORS
//
// 	    //cout<<"On node "<<(*curnode)->node_id<<endl;
//
// 	    if (((*curnode)->type)!=SENSOR) {
//
// 	      // For each incoming connection, perform adaptation based on the trait of the connection
// 	      for(curlink=((*curnode)->incoming).begin();curlink!=((*curnode)->incoming).end();++curlink) {
//
// 		if (((*curlink)->trait_id==2)||
// 		    ((*curlink)->trait_id==3)||
// 		    ((*curlink)->trait_id==4)) {
//
// 		  //In the recurrent case we must take the last activation of the input for calculating hebbian changes
// 		  if ((*curlink)->is_recurrent) {
// 		    (*curlink)->weight=
// 		      hebbian((*curlink)->weight,maxweight,
// 			      (*curlink)->in_node->last_activation,
// 			      (*curlink)->out_node->get_active_out(),
// 			      (*curlink)->params[0],(*curlink)->params[1],
// 			      (*curlink)->params[2]);
//
//
// 		  }
// 		  else { //non-recurrent case
// 		    (*curlink)->weight=
// 		      hebbian((*curlink)->weight,maxweight,
// 			      (*curlink)->in_node->get_active_out(),
// 			      (*curlink)->out_node->get_active_out(),
// 			      (*curlink)->params[0],(*curlink)->params[1],
// 			      (*curlink)->params[2]);
// 		  }
// 		}
//
// 	      }
//
// 	    }
//
// 	  }
//
// 	} //end if (adaptable)
//
// 	return true;
// }
//
// // THIS WAS NOT USED IN THE FINAL VERSION, AND NOT FULLY IMPLEMENTED,
// // BUT IT SHOWS HOW SOMETHING LIKE THIS COULD BE INITIATED
// // Note that checking networks for loops in general in not necessary
// // and therefore I stopped writing this function
// // Check Network for loops.  Return true if its ok, false if there is a loop.
// //bool Network::integrity() {
// //  std::vector<NNode*>::iterator curnode;
// //  std::vector<std::vector<NNode*>*> paths;
// //  int count;
// //  std::vector<NNode*> *newpath;
// //  std::vector<std::vector<NNode*>*>::iterator curpath;
//
// //  for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// //    newpath=new std::vector<NNode*>();
// //    paths.push_back(newpath);
// //    if (!((*curnode)->integrity(newpath))) return false;
// //  }
//
// //Delete the paths now that we are done
// //  curpath=paths.begin();
// //  for(count=0;count<paths.size();count++) {
// //    delete (*curpath);
// //    curpath++;
// //  }
//
// //  return true;
// //}
//
// // Prints the values of its outputs
// void Network::show_activation() {
// 	std::vector<NNode*>::iterator curnode;
// 	int count;
//
// 	//if (name!=0)
// 	//  cout<<"Network "<<name<<" with id "<<net_id<<" outputs: (";
// 	//else cout<<"Network id "<<net_id<<" outputs: (";
//
// 	count=1;
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 		//cout<<"[Output #"<<count<<": "<<(*curnode)<<"] ";
// 		count++;
// 	}
//
// 	//cout<<")"<<endl;
// }
//
// void Network::show_input() {
// 	std::vector<NNode*>::iterator curnode;
// 	int count;
//
// 	//if (name!=0)
// 	//  cout<<"Network "<<name<<" with id "<<net_id<<" inputs: (";
// 	//else cout<<"Network id "<<net_id<<" outputs: (";
//
// 	count=1;
// 	for(curnode=inputs.begin();curnode!=inputs.end();++curnode) {
// 		//cout<<"[Input #"<<count<<": "<<(*curnode)<<"] ";
// 		count++;
// 	}
//
// 	//cout<<")"<<endl;
// }
//
// // Add an input
// void Network::add_input(NNode *in_node) {
// 	inputs.push_back(in_node);
// }
//
// // Add an output
// void Network::add_output(NNode *out_node) {
// 	outputs.push_back(out_node);
// }
//
// // Takes an array of sensor values and loads it into SENSOR inputs ONLY
// void Network::load_sensors(double *sensvals) {
// 	//int counter=0;  //counter to move through array
// 	std::vector<NNode*>::iterator sensPtr;
//
// 	for(sensPtr=inputs.begin();sensPtr!=inputs.end();++sensPtr) {
// 		//only load values into SENSORS (not BIASes)
// 		if (((*sensPtr)->type)==SENSOR) {
// 			(*sensPtr)->sensor_load(*sensvals);
// 			sensvals++;
// 		}
// 	}
// }
//
// void Network::load_sensors(const std::vector<float> &sensvals) {
// 	//int counter=0;  //counter to move through array
// 	std::vector<NNode*>::iterator sensPtr;
// 	std::vector<float>::const_iterator valPtr;
//
// 	for(valPtr = sensvals.begin(), sensPtr = inputs.begin(); sensPtr != inputs.end() && valPtr != sensvals.end(); ++sensPtr, ++valPtr) {
// 		//only load values into SENSORS (not BIASes)
// 		if (((*sensPtr)->type)==SENSOR) {
// 			(*sensPtr)->sensor_load(*valPtr);
// 			//sensvals++;
// 		}
// 	}
// }
//
//
// // Takes and array of output activations and OVERRIDES
// // the outputs' actual activations with these values (for adaptation)
// void Network::override_outputs(double* outvals) {
//
// 	std::vector<NNode*>::iterator outPtr;
//
// 	for(outPtr=outputs.begin();outPtr!=outputs.end();++outPtr) {
// 		(*outPtr)->override_output(*outvals);
// 		outvals++;
// 	}
//
// }
//
// void Network::give_name(char *newname) {
// 	char *temp;
// 	char *temp2;
// 	temp=new char[strlen(newname)+1];
// 	strcpy(temp,newname);
// 	if (name==0) name=temp;
// 	else {
// 		temp2=name;
// 		delete temp2;
// 		name=temp;
// 	}
// }
//
// // The following two methods recurse through a network from outputs
// // down in order to count the number of nodes and links in the network.
// // This can be useful for debugging genotype->phenotype spawning
// // (to make sure their counts correspond)
//
// int Network::nodecount() {
// 	int counter=0;
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
//
//         location = std::find(seenlist.begin(),seenlist.end(),(*curnode));
// 		if (location==seenlist.end()) {
// 			counter++;
// 			seenlist.push_back(*curnode);
// 			nodecounthelper((*curnode),counter,seenlist);
// 		}
// 	}
//
// 	numnodes=counter;
//
// 	return counter;
//
// }
//
// void Network::nodecounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
//             location= std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				counter++;
// 				seenlist.push_back((*curlink)->in_node);
// 				nodecounthelper((*curlink)->in_node,counter,seenlist);
// 			}
// 		}
//
// 	}
//
// }
//
// int Network::linkcount() {
// 	int counter=0;
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 		linkcounthelper((*curnode),counter,seenlist);
// 	}
//
// 	numlinks=counter;
//
// 	return counter;
//
// }
//
// void Network::linkcounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> inlinks=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
//     location = std::find(seenlist.begin(),seenlist.end(),curnode);
// 	if ((!((curnode->type)==SENSOR))&&(location==seenlist.end())) {
// 		seenlist.push_back(curnode);
//
// 		for(curlink=inlinks.begin();curlink!=inlinks.end();++curlink) {
// 			counter++;
// 			linkcounthelper((*curlink)->in_node,counter,seenlist);
// 		}
//
// 	}
//
// }
//
// // Destroy will find every node in the network and subsequently
// // delete them one by one.  Since deleting a node deletes its incoming
// // links, all nodes and links associated with a network will be destructed
// // Note: Traits are parts of genomes and not networks, so they are not
// //       deleted here
// void Network::destroy() {
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	// Erase all nodes from all_nodes list
//
// 	for(curnode=all_nodes.begin();curnode!=all_nodes.end();++curnode) {
// 		delete (*curnode);
// 	}
//
//
// 	// -----------------------------------
//
// 	//  OLD WAY-the old way collected the nodes together and then deleted them
//
// 	//for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
// 	//cout<<seenstd::vector<<endl;
// 	//cout<<curnode<<endl;
// 	//cout<<curnode->node_id<<endl;
//
// 	//  location=find(seenlist.begin(),seenlist.end(),(*curnode));
// 	//  if (location==seenlist.end()) {
// 	//    seenlist.push_back(*curnode);
// 	//    destroy_helper((*curnode),seenlist);
// 	//  }
// 	//}
//
// 	//Now destroy the seenlist, which is all the NNodes in the network
// 	//for(curnode=seenlist.begin();curnode!=seenlist.end();++curnode) {
// 	//  delete (*curnode);
// 	//}
// }
//
// void Network::destroy_helper(NNode *curnode,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
//             location = std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				seenlist.push_back((*curlink)->in_node);
// 				destroy_helper((*curlink)->in_node,seenlist);
// 			}
// 		}
//
// 	}
//
// }
//
// // This checks a POTENTIAL link between a potential in_node and potential out_node to see if it must be recurrent
// bool Network::is_recur(NNode *potin_node,NNode *potout_node,int &count,int thresh) {
// 	std::vector<Link*>::iterator curlink;
//
// 	++count;  //Count the node as visited
//
// 	if (count>thresh) {
// 		//cout<<"returning false"<<endl;
// 		return true;  //Short out the whole thing- loop detected
// 	}
//
// 	if (potin_node==potout_node) return true;
// 	else {
// 		//Check back on all links...
// 		for(curlink=(potin_node->incoming).begin();curlink!=(potin_node->incoming).end();curlink++) {
// 			//But skip links that are already recurrent
// 			//(We want to check back through the forward flow of signals only
// 			if (!((*curlink)->is_recurrent)) {
// 				if (is_recur((*curlink)->in_node,potout_node,count,thresh)) return true;
// 			}
// 		}
// 		return false;
// 	}
// }
//
// bool Network::is_recur2(NNode *potin_node,NNode *potout_node,int &count, int thresh) {
// 	NNode* curnode;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
// 	curnode=potin_node; //start at out node, if we find a path that lands at in_node, then this link will be recurrent
// 	seenlist.push_back(curnode);
// 	return is_rec_helper(curnode,potout_node,seenlist);
// }
//
// bool Network::is_rec_helper(NNode* curnode, NNode* find_node,std::vector<NNode*> &seenlist)
// {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
// 			if((*curlink)->in_node==find_node)
// 				return true;
//             location= std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				seenlist.push_back((*curlink)->in_node);
// 				if(is_rec_helper((*curlink)->in_node,find_node,seenlist))
// 					return true;
// 			}
// 		}
// 	}
// 	return false;
// }
//
// #<{(|
// int Network::nodecount() {
// 	int counter=0;
// 	std::vector<NNode*>::iterator curnode;
// 	std::vector<NNode*>::iterator location;
// 	std::vector<NNode*> seenlist;  //List of nodes not to doublecount
//
// 	for(curnode=outputs.begin();curnode!=outputs.end();++curnode) {
//
//         location = std::find(seenlist.begin(),seenlist.end(),(*curnode));
// 		if (location==seenlist.end()) {
// 			counter++;
// 			seenlist.push_back(*curnode);
// 			nodecounthelper((*curnode),counter,seenlist);
// 		}
// 	}
//
// 	numnodes=counter;
//
// 	return counter;
//
// }
//
// void Network::nodecounthelper(NNode *curnode,int &counter,std::vector<NNode*> &seenlist) {
// 	std::vector<Link*> innodes=curnode->incoming;
// 	std::vector<Link*>::iterator curlink;
// 	std::vector<NNode*>::iterator location;
//
// 	if (!((curnode->type)==SENSOR)) {
// 		for(curlink=innodes.begin();curlink!=innodes.end();++curlink) {
//             location= std::find(seenlist.begin(),seenlist.end(),((*curlink)->in_node));
// 			if (location==seenlist.end()) {
// 				counter++;
// 				seenlist.push_back((*curlink)->in_node);
// 				nodecounthelper((*curlink)->in_node,counter,seenlist);
// 			}
// 		}
//
// 	}
//
// }
// |)}>#
//
// int Network::input_start() {
// 	input_iter=inputs.begin();
// 	return 1;
// }
//
// int Network::load_in(double d) {
// 	(*input_iter)->sensor_load(d);
// 	input_iter++;
// 	if (input_iter==inputs.end()) return 0;
// 	else return 1;
// }
//
//
// //Find the maximum number of neurons between an ouput and an input
// int Network::max_depth() {
//   std::vector<NNode*>::iterator curoutput; //The current output we are looking at
//   int cur_depth; //The depth of the current node
//   int max=0; //The max depth
//   int count=0; //a way to prevent infinite loops
//   int thresh=(all_nodes.size())*(all_nodes.size());
//
//   for(curoutput=outputs.begin();curoutput!=outputs.end();curoutput++) {
//     cur_depth=(*curoutput)->depth(0,this,count,thresh);
//     if (cur_depth>max) max=cur_depth;
//   }
//
//   return max;
//
// }
//