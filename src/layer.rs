use crate::neuron::Neuron;
use crate::value::TensorId;

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_in: i32, n_out: i32) -> Layer {
        let mut neurons: Vec<Neuron> = vec![];
        for _ in 0..n_out {
            neurons.push(Neuron::new(n_in));
        }

        Layer { neurons }
    }

    pub fn forward(&self, x: &Vec<TensorId>) -> Vec<TensorId> {
        let mut outs: Vec<TensorId> = vec![];
        for i in 0..self.neurons.len() {
            let n = &self.neurons[i];
            outs.push(n.forward(x));
        }

        outs
    }
}
