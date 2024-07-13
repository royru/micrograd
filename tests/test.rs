use micrograd::Value;
use micrograd::MLP;

#[test]
fn test_train() {
    // watch the test output:
    // cargo test test_train -- --show-output
    let mlp = MLP::new(&vec![2, 4, 1]);
    let x = vec![2.0, 3.0];
    let y = -1.0;

    // first prediction
    let p1 = mlp.predict(&x);

    for _ in 0..20 {
        mlp.train(&x, y);
    }

    // second prediction
    let p2 = mlp.predict(&x);

    // p2 should be smaller, as the train steps should get closer to y=-1.0:
    assert!(p1[0] > p2[0]);
}

#[test]
fn test_forward() {
    let a = Value::val(3.0);
    let b = Value::val(2.0);
    assert_eq!(a.mul(&b).get_val(), 6.0);
}

#[test]
fn test_backward() {
    let a = Value::val(3.0);
    let b = Value::val(2.0);
    let c = b.mul(&a);
    c.backwards();
    assert_eq!(a.get_grad(), 2.0);
}
