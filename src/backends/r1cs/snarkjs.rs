
use ark_bls12_381::Fr;
use ark_ff::fields::PrimeField;

use constraint_writers::r1cs_writer::{ConstraintSection, HeaderData, R1CSWriter};
use itertools::Itertools;
use crate::error::Result;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom};
use std::io::{BufWriter, Write};
use std::vec;

// use ark_ff::BigInteger;

use num_bigint_dig::BigInt;
use super::{GeneratedWitness, LinearCombination, R1csBls12_381};

#[derive(Debug)]
struct SnarkjsConstraint {
    pub a: SnarkjsLinearCombination,
    pub b: SnarkjsLinearCombination,
    pub c: SnarkjsLinearCombination,
}

#[derive(Debug)]
struct SnarkjsLinearCombination {
    pub terms: HashMap<usize, BigInt>,
    pub constant: BigInt,
}

impl SnarkjsLinearCombination {
    fn to_hashmap(&self) -> HashMap<usize, BigInt> {
        let mut terms = self.terms.clone();
        
        // add the constant term with var indexed at 0
        if terms.insert(0, self.constant.clone()).is_some() {
            // sanity check
            panic!("The first var should be preserved for constant term");
        }

        terms
    }
}

/// calculate the number of bytes for the prime field
fn field_size(prime: &BigInt) -> usize {
    if prime.bits() % 64 == 0 {
        prime.bits() / 8
    } else {
        (prime.bits() / 64 + 1) * 8
    }
}

/// A struct to export r1cs circuit and witness to the snarkjs formats
pub struct SnarkjsExporter {
    /// A R1CS backend with the circuit finalized
    r1cs_backend: R1csBls12_381,
    /// A mapping between the witness vars' indexes in the backend and the new indexes arranged for the snarkjs format.
    /// <original, new>: The key (left) is the original index of the witness var in the backend, and the value (right) is the new index.
    /// This mapping is used to re-arrange the witness values to align with the snarkjs format.
    /// - The format assumes the public outputs and inputs are at the beginning of the witness vector.
    /// - The variables in the constraints needs this mapping to reference to the new witness vector.
    witness_map: HashMap<usize, usize>,
}

impl SnarkjsExporter {
    /// During the initialization, the witness map is created to re-arrange the witness vector.
    /// The reordering of witness vars:
    /// 1. The first var is always reserved and valued as 1.
    /// 2. The public outputs are stacked first.
    /// 3. The public inputs are stacked next.
    /// 4. The rest of the witness vars are stacked last.
    pub fn new (r1cs_backend: R1csBls12_381) -> SnarkjsExporter {
        let mut witness_map = HashMap::new();
        
        let mut witness_vars = r1cs_backend.witness_vars.clone();

        // group all the public items together
        // outputs are intended to be before inputs
        let public_items = r1cs_backend.public_outputs.iter().chain(r1cs_backend.public_inputs.iter());

        for (index, var) in public_items.enumerate() {
            // first var is fixed, so here we start from 1
            witness_map.insert(var.index, index + 1);

            // remove the public input from the witness vars
            witness_vars.remove(&var.index);
        }

        // stack in the rest of the witness vars
        for index in witness_vars.keys().sorted() {
            witness_map.insert(*index, witness_map.len() + 1);
        }

        // witness_map should have all the witness vars
        assert_eq!(r1cs_backend.witness_vars.len(), witness_map.len());

        SnarkjsExporter {
            r1cs_backend,
            witness_map,
        }
    }

    /// Restructure the linear combination to align with the snarkjs format
    /// - use witness mapper to re-arrange the variables
    /// - convert the factors to BigInt
    fn restructure_lc(&self, lc: &LinearCombination) -> SnarkjsLinearCombination {
        let terms = lc.terms.iter().map(|(cvar, factor)| {
            let new_index: usize = *self.witness_map.get(&cvar.index).unwrap();
            let factor_bigint = Self::convert_to_bigint(factor);

            (new_index, factor_bigint)
        }).collect();

        let constant = Self::convert_to_bigint(&lc.constant);

        SnarkjsLinearCombination { terms, constant }
    }

    /// Restructure the constraints to align with the snarkjs format
    fn restructure_constraints(&self) -> Vec<SnarkjsConstraint> {
        let mut constraints = vec![];
        for constraint in &self.r1cs_backend.constraints {
            constraints.push(SnarkjsConstraint {
                a: self.restructure_lc(&constraint.a),
                b: self.restructure_lc(&constraint.b),
                c: self.restructure_lc(&constraint.c),
            });
        }
        constraints
    }

    /// Restructure the witness vector
    /// 1. add the first var that is always valued as 1
    /// 2. use witness mapper to re-arrange the variables
    /// 3. convert the witness values to BigInt
    /// 4. convert to a witness vector ordered by new index
    fn restructure_witness(&self, generated_witness: GeneratedWitness) -> Vec<BigInt> {
        let mut restructured_witness_values = HashMap::new();

        // add the first var that is always valued as 1
        restructured_witness_values.insert(0, BigInt::from(1));

        for (id, value) in generated_witness.witness.iter() {
            let new_index = self.witness_map.get(id).unwrap();
            let value_bigint = Self::convert_to_bigint(value);

            restructured_witness_values.insert(*new_index, value_bigint);
        }

        // convert to vector ordered by new index
        restructured_witness_values
            .iter()
            .sorted_by(|a, b| a.0.cmp(b.0))
            .map(|x| x.1.clone())
            .collect::<Vec<_>>()
    }
    
    /// Generate the r1cs file in snarkjs format
    /// It uses the circom rust library to generate the r1cs file.
    /// The binary format spec: https://github.com/iden3/r1csfile/blob/master/doc/r1cs_bin_format.md
    pub fn gen_r1cs_file(&self, file: &str) {
        let prime = &self.r1cs_backend.prime;
        let field_size = field_size(prime);

        let r1cs = R1CSWriter::new(file.to_string(), field_size, false).unwrap();
        let mut constraint_section = R1CSWriter::start_constraints_section(r1cs).unwrap();

        let restructure_constraints = self.restructure_constraints();

        for constraint in &restructure_constraints {
            ConstraintSection::write_constraint_usize(
                &mut constraint_section,
                &constraint.a.to_hashmap(),
                &constraint.b.to_hashmap(),
                &constraint.c.to_hashmap(),
            );
        }

        let r1cs = constraint_section.end_section().unwrap();
        let mut header_section = R1CSWriter::start_header_section(r1cs).unwrap();
        let header_data = HeaderData {
            // Snarkjs uses this meta data to determine which curve to use
            field: prime.clone(),
            // Both the sizes of public inputs and outputs will be crucial for the snarkjs to determine 
            // where to extract the public inputs and outputs from the witness vector.
            // Snarkjs prove command will generate a public.json file that contains the public inputs and outputs.
            // So if the sizes are not correct, the public.json will not be generated correctly, which might potential contains the private inputs.
            public_outputs: self.r1cs_backend.public_outputs.len(),
            public_inputs: self.r1cs_backend.public_inputs.len(),
            // There seems no use of this field in the snarkjs lib. It might be just a reference.
            private_inputs: self.r1cs_backend.private_input_number(),
            // Add one to take into account the first var that is only added during the witness formation for snarkjs
            total_wires: self.witness_map.len() + 1,
            // This is for circom lang debugging, so we don't need it
            number_of_labels: 0,
            number_of_constraints: restructure_constraints.len(),
        };
        header_section.write_section(header_data);

        let r1cs = header_section.end_section().unwrap();
        R1CSWriter::finish_writing(r1cs);
    }

    /// Generate the wtns file in snarkjs format
    pub fn gen_wtns_file(&self, file: &str, witness: GeneratedWitness) {
        let restructured_witness = self.restructure_witness(witness);

        let mut witness_writer = WitnessWriter::new(file).unwrap();

        witness_writer.write(restructured_witness, &self.r1cs_backend.prime);
    }

    fn convert_to_bigint(value: &Fr) -> BigInt {
        BigInt::from_bytes_le(
            num_bigint_dig::Sign::Plus,
            &ark_ff::BigInteger::to_bytes_le(&value.into_repr()),
        )
    }
}


/// Witness writer for generating the wtns file in snarkjs format
/// The circom rust lib seems not to have the API to generate the wtns file, so we create our own based on the snarkjs lib.
/// The implementation follows: https://github.com/iden3/snarkjs/blob/577b3f358016a486402050d3b7242876082c085f/src/wtns_utils.js#L25
/// It uses the same binary format as the r1cs file relies on. 
/// Although it is the same binary file protocol, but the witness file has simpler data points than the r1cs file.
struct WitnessWriter {
    inner: BufWriter<File>,
    writing_section: Option<WritingSection>,
    section_size_position: u64,
}

/// A struct to represent a section in the snarkjs binary format
struct WritingSection;

impl WitnessWriter {
    // Initialize a FileWriter
    pub fn new(
        path: &str,
    ) -> Result<WitnessWriter> {
        // file type for the witness file
        let file_type = "wtns";
        // version of the file format
        let version = 2u32;
        // total number of sections
        let n_sections = 2u32;

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .unwrap();
        let mut writer = BufWriter::new(file);

        // Write the file type (magic string) as bytes
        let file_type_bytes = file_type.as_bytes();
        if file_type_bytes.len() != 4 {
            panic!("File type must be 4 characters long");
        }
        writer.write_all(file_type_bytes);

        // Write the version as a 32-bit unsigned integer in little endian
        writer.write_all(&version.to_le_bytes());

        // Write the number of sections as a 32-bit unsigned integer in little endian
        writer.write_all(&n_sections.to_le_bytes());

        let current_position = writer.stream_position().unwrap();

        Ok(WitnessWriter {
            inner: writer,
            writing_section: None,
            section_size_position: current_position,
        })
    }

    /// Start a new section for writing
    pub fn start_write_section(&mut self, id_section: u32) {
        // Write the section ID as ULE32
        self.inner.write_all(&id_section.to_le_bytes()); 
        // Get the current position
        self.section_size_position = self.inner.stream_position().unwrap(); 
        // Temporarily write 0 as ULE64 for the section size
        self.inner.write_all(&0u64.to_le_bytes()); 
        self.writing_section = Some(WritingSection);
    }

    /// End the current section
    pub fn end_write_section(&mut self) {
        let current_pos = self.inner.stream_position().unwrap();
        // Calculate the size of the section
        let section_size = current_pos - self.section_size_position - 8; 

        // Move back to where the size needs to be written
        self.inner.seek(SeekFrom::Start(self.section_size_position)); 
        // Write the actual section size
        self.inner.write_all(&section_size.to_le_bytes()); 
        // Return to the end of the section
        self.inner.seek(SeekFrom::Start(current_pos)); 
        // Flush the buffer to ensure all data is written to the file
        self.inner.flush(); 

        self.writing_section = None;
    }

    /// Write the witness to the file
    /// It stores the two sections:
    /// - Header section: describes the field size, prime field, and the number of witnesses
    /// - Witness section: contains the witness values
    pub fn write(&mut self, witness: Vec<BigInt>, prime: &BigInt) {
        // Start writing the first section
        self.start_write_section(1);
        // Write field size in number of bytes
        let field_n_bytes = field_size(prime);
        self.inner.write_all(&(field_n_bytes as u32).to_le_bytes());
        // Write the prime field in bytes
        self.write_big_int(prime.clone(), field_n_bytes);
        // Write the number of witnesses
        self.inner.write_all(&(witness.len() as u32).to_le_bytes());
        
        self.end_write_section();

        // Start writing the second section
        self.start_write_section(2);

        /// Write the witness values to the file
        /// Each witness value occupies the same number of bytes as the prime field
        for value in witness {
            self.write_big_int(value.clone(), field_n_bytes as usize);
        }
        self.end_write_section();
    }

    /// Write a BigInt to the file
    fn write_big_int(&mut self, value: BigInt, size: usize) {
        let bytes = value.to_bytes_le().1;

        let mut buffer = vec![0u8; size];
        buffer[..bytes.len()].copy_from_slice(&bytes);
        self.inner.write_all(&buffer);
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use num_bigint_dig::BigInt;

    use crate::{backends::{r1cs::R1csBls12_381, Backend}, constants::Span, var::Value, witness::WitnessEnv};

    #[test]
    fn test_restructure_witness() {
        let mut r1cs = R1csBls12_381::new();

        // mock a constraint
        let span = Span::default();
        let var1_val = 2;
        let public_input_val = 3;
        let sum_val = var1_val + public_input_val;
        let var1 = r1cs.new_internal_var(Value::Constant(Fr::from(var1_val)), span);
        let public_input_var = r1cs.add_public_input(Value::Constant(Fr::from(public_input_val)), span);
        let sum_var = r1cs.add(&var1, &public_input_var, span);
        r1cs.add_public_output(Value::PublicOutput(Some(sum_var)), span);
        let public_output_val = sum_val;

        // convert witness to snarkjs format
        let mut snarkjs_exporter = super::SnarkjsExporter::new(r1cs);
        assert_eq!(snarkjs_exporter.witness_map.len(), snarkjs_exporter.r1cs_backend.witness_vars.len());

        // mock finalizing the circuit
        snarkjs_exporter.r1cs_backend.finalized = true;
        let mut witness_env = WitnessEnv::default();
        let generated_witness = snarkjs_exporter.r1cs_backend.generate_witness(&mut witness_env);

        let rearranged_witness = snarkjs_exporter.restructure_witness(generated_witness.unwrap());

        assert_eq!(
            rearranged_witness,
            vec![
                // first var is always 1
                BigInt::from(1),
                // output ordered before input
                BigInt::from(public_output_val),
                // public input 
                BigInt::from(public_input_val),
                // private inputs (the rest of the witness vars)
                BigInt::from(var1_val),
                BigInt::from(sum_val),
            ]
        );

        // instead, maybe refactor this test to check if the constraits are evaluated to zero with the reordered witness?
        // so we just check the accrued results (to simplify this test)
        let restructure_constraints = snarkjs_exporter.restructure_constraints();
        for (rc, oc) in restructure_constraints.iter().zip(snarkjs_exporter.r1cs_backend.constraints.iter()) {
            // concat the terms from a b c
            let all_original_terms = [oc.a.terms.clone(), oc.b.terms.clone(), oc.c.terms.clone()];
            let all_restructured_terms = [rc.a.terms.clone(), rc.b.terms.clone(), rc.c.terms.clone()];

            for (oterms, rterms) in all_original_terms.iter().zip(all_restructured_terms) {
                assert_eq!(oterms.len(), rterms.len());
                for original_var in oterms.keys() {
                    // check if the original var is in the restructured terms after reordering
                    let new_index = snarkjs_exporter.witness_map.get(&original_var.index).unwrap();
                    assert!(rterms.contains_key(new_index));
                }
            }
        }
    }
}