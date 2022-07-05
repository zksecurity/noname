use num_bigint::BigUint;

pub type Field = kimchi::mina_curves::pasta::Fp;

pub trait PrettyField: ark_ff::PrimeField {
    fn pretty(&self) -> String {
        let bigint: BigUint = (*self).into();
        let inv: BigUint = self.neg().into(); // gettho way of splitting the field into positive and negative elements
        if inv < bigint {
            format!("-{}", inv)
        } else {
            bigint.to_string()
        }
    }
}

impl PrettyField for Field {}
