use crate::layer::Layer;
use crate::value::{TensorId, Value};

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(dims: &Vec<i32>) -> MLP {
        let mut layers: Vec<Layer> = vec![];
        let l = dims.len();
        for i in 0..l - 1 {
            layers.push(Layer::new(dims[i], dims[i + 1]))
        }

        MLP { layers }
    }

    pub fn predict(&self, x: &Vec<f32>) -> Vec<f32> {
        let x = x.iter().map(|v| Value::val(*v)).collect();
        let out = self.pred(&x);
        out.iter().map(|v| v.get_val()).collect()
    }

    fn pred(&self, x: &Vec<TensorId>) -> Vec<TensorId> {
        let mut out = x.clone();
        for i in 0..self.layers.len() {
            let l = &self.layers[i];
            out = l.forward(&out);
        }
        out
    }

    pub fn train(&self, x: &Vec<f32>, y: f32) {
        // turn floats into Values
        let x = x.iter().map(|xi| Value::val(*xi)).collect();
        let y_true = Value::val(y);
        // do the prediction...
        let y_pred = self.pred(&x);
        // (y_true - y_pred)**2
        let diff = y_pred[0].sub(&y_true);
        let loss = diff.mul(&diff);

        // calculate gradients
        loss.backwards();

        println!("loss: {}, y_pred: {}", loss.get_val(), y_pred[0].get_val());

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            for n in 0..layer.neurons.len() {
                let neuron = &layer.neurons[n];

                // update weights
                for w in 0..neuron.weights.len() {
                    let weight = neuron.weights[w];
                    weight.set_val(weight.get_val() + (-0.01 * weight.get_grad()));
                    weight.set_grad(0.0); // reset gradient
                }

                // update bias
                let bias = neuron.bias;
                bias.set_val(bias.get_val() + (-0.01 * bias.get_grad()));
                bias.set_grad(0.0);
            }
        }
    }
}
