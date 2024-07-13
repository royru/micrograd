# micrograd
As inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and the [lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) about it.

```rust 
use micrograd::Value;

let a = Value::val(3.0);
let b = Value::val(2.0);
let c = b.mul(&a);
c.backwards();

assert_eq!(a.get_grad(), 2.0);
```

You can also define a MLP and train it:

```rust
use micrograd::MLP;

let mlp = MLP::new(&vec![2, 4, 1]);
let x = vec![2.0, 3.0];
let y = -1.0;

for _ in 0..20 {
  mlp.train(&x, y);
}

mlp.predict(&x) // hopefully close to -1
```