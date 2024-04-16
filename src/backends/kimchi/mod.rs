use crate::constants::Field;

use super::Backend;

#[derive(Clone)]
pub struct KimchiVesta;

impl Backend for KimchiVesta {
    type Field = Field;
    
    fn poseidon() -> crate::imports::FnHandle<Self> {
        todo!()
    }
}