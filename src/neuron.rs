use crate::{value::TensorId, value::Value};
use rand::Rng;

pub struct Neuron {
    pub weights: Vec<TensorId>,
    pub bias: TensorId,
}

impl Neuron {
    pub fn new(n_in: i32) -> Neuron {
        let mut weights: Vec<TensorId> = vec![];
        for _ in 0..n_in {
            let v = rand::thread_rng().gen_range(-1.0..1.0);
            let weight = Value::val(v);
            weights.push(weight);
        }

        let v = rand::thread_rng().gen_range(-1.0..1.0);
        let bias = Value::val(v);
        Neuron { weights, bias }
    }

    pub fn forward(&self, x: &Vec<TensorId>) -> TensorId {
        // w * x + b
        assert!(x.len() == self.weights.len(), "vector lengths don't match");

        // + b
        let mut out = self.bias;

        for i in 0..x.len() {
            // sum += wi * xi;
            let wi = self.weights[i];
            let xi = x[i];
            let mul = wi.mul(&xi);
            out = mul.add(&out);
        }

        // activation fn
        out.tanh()
    }
}
