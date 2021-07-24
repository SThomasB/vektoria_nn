
use vektoria::*;

pub fn init_layer<T,F>(inputs: usize, outputs: usize , init_funct: F ) -> Vec<Vec<T>> 
    where 
        T: SkalarType,
        F: Fn(T) -> T {
            vec![vec![T::from(0).unwrap(); outputs]; inputs+1]
                .iter()
                .map(|a| a.iter().map(|b| init_funct(*b)).collect())
                .collect()
}

type Matrise = Vec<Vec<f64>>;
type Tensor<'a>  = Vec<&'a Matrise>;
type FunList = Vec<Box<dyn Fn(&f64) -> f64>>;
pub struct Network<'a> {
    pub layers: Tensor<'a>,
    pub activations: FunList,
}

impl <'a>Network<'a> {
    pub fn new(layers: Tensor<'a>, activations: Vec<Box<dyn Fn(&f64) -> f64>>) -> Network<'a> {
        Network{
            layers: layers,
            activations: activations
        }
    }
    pub fn forward(&self, input: Vec<f64>) -> Vec<f64> {
        self.layers.iter()
                    .zip(self.activations.iter())
                    .fold(input, |a,b| a.in_augmented(b.0).iter()
                    .map(b.1).collect::<Vec<f64>>())
    }
}

    
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        use super::*;
        // implementing one layer
        let x = vec![1.0; 4]; // 3 inputs plus bias
        let l1 = init_layer(3, 4, |_| 1.0);
        let l2 = init_layer(4, 3, |_| 1.0);
        let f1 = |a: &f64| a.to_owned();
        let fs : Vec<Box<dyn Fn(&f64)-> f64>> = vec![Box::new(f1), Box::new(f1)];
        let net = Network::new(vec![&l1, &l2], fs);

        assert_eq!(net.forward(vec![1.0,1.0,1.0]), vec![17.0,17.0,17.0]);
    }
}
