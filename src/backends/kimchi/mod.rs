use crate::constants::Field;

use super::Backend;
pub mod builtin;

#[derive(Clone)]
pub struct KimchiVesta;

impl Backend for KimchiVesta {
    type Field = Field;
    
    fn poseidon() -> crate::imports::FnHandle<Self> {
        builtin::poseidon
    }
}