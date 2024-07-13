use std::collections::{HashMap, HashSet, VecDeque};

use lazy_static::lazy_static;
use std::sync::Mutex;

// https://stackoverflow.com/questions/27791532/how-do-i-create-a-global-mutable-singleton
lazy_static! {
    static ref DATA: Mutex<HashMap<TensorId, Value>> = Mutex::new(HashMap::new());
}

// unique identifier for tensors
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Empty,
    Mul,
    Sub,
    Add,
    Tanh,
}

pub struct Value {
    id: TensorId,
    data: f32,
    children: (Option<TensorId>, Option<TensorId>),
    op: Op,
    grad: f32,
}

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn get_grad(&self) -> f32 {
        let map = DATA.lock().unwrap();
        let val = map.get(&self).expect("node should exist");
        val.grad
    }

    pub fn set_grad(&self, grad: f32) {
        let mut map = DATA.lock().unwrap();
        let val = map.get_mut(&self).expect("node should exist");
        val.grad = grad;
    }

    pub fn get_val(&self) -> f32 {
        let map = DATA.lock().unwrap();
        let val = map.get(&self).expect("node should exist");
        val.data
    }

    pub fn set_val(&self, v: f32) {
        let mut map = DATA.lock().unwrap();
        let val = map.get_mut(&self).expect("node should exist");
        val.data = v;
    }

    pub fn mul(&self, k: &TensorId) -> TensorId {
        let mut map = DATA.lock().unwrap();
        let val1 = map.get(&self).expect("node should exist");
        let val2 = map.get(&k).expect("node should exist");
        let out = Value::new(
            val1.data * val2.data,
            (Some(val1.id), Some(val2.id)),
            Op::Mul,
        );
        let id = out.id;
        map.insert(id, out);
        id
    }

    pub fn sub(&self, k: &TensorId) -> TensorId {
        let mut map = DATA.lock().unwrap();
        let val1 = map.get(&self).expect("node should exist");
        let val2 = map.get(&k).expect("node should exist");
        let out = Value::new(
            val1.data - val2.data,
            (Some(val1.id), Some(val2.id)),
            Op::Sub,
        );
        let id = out.id;
        map.insert(id, out);
        id
    }

    pub fn add(&self, k: &TensorId) -> TensorId {
        let mut map = DATA.lock().unwrap();
        let val1 = map.get(&self).expect("node should exist");
        let val2 = map.get(&k).expect("node should exist");
        let out = Value::new(
            val1.data + val2.data,
            (Some(val1.id), Some(val2.id)),
            Op::Add,
        );
        let id = out.id;
        map.insert(id, out);
        id
    }

    pub fn tanh(&self) -> TensorId {
        let mut map = DATA.lock().unwrap();
        let val = map.get(&self).expect("node should exist");
        let out = Value::new(val.data.tanh(), (Some(val.id), None), Op::Tanh);
        let id = out.id;
        map.insert(id, out);
        id
    }

    pub fn backwards(&self) {
        // initialize root gradient
        self.set_grad(1.0);

        // build topology
        let mut visited: HashSet<TensorId> = HashSet::new();
        let mut topo: Vec<TensorId> = Vec::new();
        self.build_topo(&mut visited, &mut topo);

        for id in topo.iter().rev() {
            id.backward();
        }
    }

    fn build_topo(&self, visited: &mut HashSet<TensorId>, topo: &mut Vec<TensorId>) {
        let mut stack = VecDeque::new();
        stack.push_back((*self, false)); // (id, children_visited)

        // while instead of recursion: https://stackoverflow.com/questions/65948553/why-is-recursion-not-suggested-in-rust
        while let Some((node, children_visited)) = stack.pop_back() {
            if !visited.contains(&node) {
                visited.insert(node);

                if !children_visited {
                    stack.push_back((node, true)); // Mark this node as children visited in the future
                    let map = DATA.lock().unwrap();
                    if let Some(value) = map.get(&node) {
                        if value.children.0.is_some() {
                            stack.push_back((value.children.0.unwrap(), false));
                        }
                        if value.children.1.is_some() {
                            stack.push_back((value.children.1.unwrap(), false));
                        }
                    } else {
                        panic!("value must exist")
                    }
                } else {
                    topo.push(node);
                }
            } else if children_visited {
                topo.push(node);
            }
        }
    }

    // OTODO: needs refactoring
    fn backward(&self) {
        let op;
        let children;
        let grad;
        let data;
        let mut map = DATA.lock().unwrap();

        if let Some(value) = map.get_mut(self) {
            op = value.op;
            children = value.children;
            grad = value.grad;
            data = value.data;
        } else {
            panic!("value must exist")
        }

        if op == Op::Mul {
            let left_id = children.0.expect("lefthand must exist");
            let right_id = children.1.expect("righthand must exist");
            let mut left_data = 0.0;
            let mut right_data = 0.0;

            if let Some(left) = map.get_mut(&left_id) {
                left_data = left.data;
            }

            if let Some(right) = map.get_mut(&right_id) {
                right_data = right.data;
                right.grad += left_data * grad;
            }

            if let Some(left) = map.get_mut(&left_id) {
                left.grad += right_data * grad;
            }
        }

        if op == Op::Add || op == Op::Sub {
            let left_id = children.0.expect("lefthand must exist");
            let right_id = children.1.expect("righthand must exist");

            if let Some(left) = map.get_mut(&left_id) {
                left.grad += grad;
            }

            if let Some(right) = map.get_mut(&right_id) {
                right.grad += grad;
            }
        }

        if op == Op::Tanh {
            let id = children.0.expect("child must exist");
            if let Some(val) = map.get_mut(&id) {
                val.grad += (1.0 - data.powi(2)) * grad;
            }
        }
    }
}

impl Value {
    fn new(data: f32, children: (Option<TensorId>, Option<TensorId>), op: Op) -> Value {
        Value {
            id: TensorId::new(),
            data,
            children,
            op,
            grad: 0.0,
        }
    }

    pub fn val(v: f32) -> TensorId {
        let val = Value::new(v, (None, None), Op::Empty);
        let id = val.id;
        DATA.lock().unwrap().insert(id, val);
        id
    }
}
