use vektoria_nn::*;
use vektoria::*;
use rand::prelude::*;
fn main() {

let f = |a: &f64| {a.max(0.0)};
let g = |a: &f64| {1.0/(1.0+(-a).exp())};
let mu = vec![100,255,255,255];
let t1 = std::time::Instant::now();
let l1 = init_layer(6,4, |_| thread_rng().gen::<f64>());
let l2 = init_layer(4,3, |_| thread_rng().gen::<f64>());
let l3 = init_layer(3,2, |_| thread_rng().gen::<f64>());
println!("Time to produce random layers: {}{}s", t1.elapsed().as_micros(), '\u{03BC}');
let n = vec![&l1,&l2,&l3];
let fs: Vec<Box<dyn Fn(&f64) -> f64>> = vec![Box::new(f),Box::new(f), Box::new(g)];
let net = Network::new(n,fs);


let t2 = std::time::Instant::now();
let v_to_print = net.forward(vec![1.22,0.01,10.0,1.0]);
println!("feedforward: {}\u{03BC}s", t2.elapsed().as_micros());         
println!("{:#?}", v_to_print);
}