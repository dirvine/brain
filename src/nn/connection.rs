struct Connection {
    unsigned short int m_source_neuron_idx;       // index of source neuron
    unsigned short int m_target_neuron_idx;       // index of target neuron
    double m_weight;                               // weight of the connection
    double m_signal;                               // weight * input signal

    bool m_recur_flag; // recurrence flag for displaying purposes
    // can be ignored

    // Hebbian learning parameters
    // Ignored in case there is no lifetime learning
    double m_hebb_rate;
    double m_hebb_pre_rate;

    // comparison operator (nessesary for boost::python)
    bool operator==(Connection const& other) const
    {
        if ((m_source_neuron_idx == other.m_source_neuron_idx) &&
            (m_target_neuron_idx == other.m_target_neuron_idx)) /*&&
            (m_weight == other.m_weight) &&
            (m_recur_flag == other.m_recur_flag))*/
            return true;
        else
            return false;
    }

}

