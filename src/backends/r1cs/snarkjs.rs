use crate::backends::BackendField;
use constraint_writers::r1cs_writer::{ConstraintSection, HeaderData, R1CSWriter};
use miette::Diagnostic;
use thiserror::Error;

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::vec;

use super::{GeneratedWitness, LinearCombination, R1CS};
use num_bigint_dig::BigInt;

#[derive(Diagnostic, Debug, Error)]
pub enum Error {
    /// An error type associated with [`R1CSWriter`].
    ///
    /// It turns out that even [`R1CSWriter`] fails to write to file,
    /// the error is ignored and a unit type is returned,
    /// so we come up with a custom type to represent that.
    #[error("Something went wrong writing the R1CS file")]
    R1CSWriterIo,
    /// An error type associated with [`WitnessWriter`].
    #[error(transparent)]
    WitnessWriterIo(#[from] std::io::Error),
}

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
        // sanity check
        assert!(
            terms.insert(0, self.constant.clone()).is_none(),
            "The first var should be preserved for constant term"
        );

        terms
    }
}

/// Calculate the number of bytes for the prime field.
fn field_size(prime: &BigInt) -> usize {
    if prime.bits() % 64 == 0 {
        prime.bits() / 8
    } else {
        (prime.bits() / 64 + 1) * 8
    }
}

/// A struct to export r1cs circuit and witness to the snarkjs formats.
pub struct SnarkjsExporter<F>
where
    F: BackendField,
{
    /// A R1CS backend with the circuit finalized.
    r1cs_backend: R1CS<F>,
}

// TODO: The impls in this file return a `noname::error::Result`, but writers return
// std::io::Error. Remove this allowance once we properly handle errors.
#[allow(unused_results)]
impl<F> SnarkjsExporter<F>
where
    F: BackendField,
{
    #[must_use]
    pub fn new(r1cs_backend: R1CS<F>) -> SnarkjsExporter<F> {
        SnarkjsExporter { r1cs_backend }
    }

    /// Restructure the linear combination to align with the snarkjs format.
    /// - convert the factors to `BigInt`.
    fn restructure_lc(&self, lc: &LinearCombination<F>) -> SnarkjsLinearCombination {
        let terms = lc
            .terms
            .iter()
            .map(|(cvar, factor)| {
                let factor_bigint = Self::convert_to_bigint(factor);

                (cvar.index, factor_bigint)
            })
            .collect();

        let constant = Self::convert_to_bigint(&lc.constant);

        SnarkjsLinearCombination { terms, constant }
    }

    /// Restructure the constraints to align with the snarkjs format.
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
    fn restructure_witness(&self, generated_witness: GeneratedWitness<F>) -> Vec<BigInt> {
        // convert the witness to BigInt
        generated_witness
            .witness
            .iter()
            .map(|value| Self::convert_to_bigint(value))
            .collect()
    }

    /// Generate the r1cs file in snarkjs format.
    /// It uses the circom rust library to generate the r1cs file.
    /// The binary format spec: https://github.com/iden3/r1csfile/blob/master/doc/r1cs_bin_format.md
    pub fn gen_r1cs_file(&self, file: &str) -> Result<(), Error> {
        let prime = self.backend_prime();
        let field_size = field_size(&prime);

        let r1cs = R1CSWriter::new(file.to_string(), field_size, false).unwrap();
        let mut constraint_section = R1CSWriter::start_constraints_section(r1cs).unwrap();

        let restructure_constraints = self.restructure_constraints();

        for constraint in &restructure_constraints {
            ConstraintSection::write_constraint_usize(
                &mut constraint_section,
                &constraint.a.to_hashmap(),
                &constraint.b.to_hashmap(),
                &constraint.c.to_hashmap(),
            )
            .map_err(|_| Error::R1CSWriterIo)?;
        }

        let r1cs = constraint_section.end_section().unwrap();
        let mut header_section = R1CSWriter::start_header_section(r1cs).unwrap();
        let header_data = HeaderData {
            // Snarkjs uses this meta data to determine which curve to use.
            field: prime,
            // Both the sizes of public inputs and outputs will be crucial for the snarkjs to determine
            // where to extract the public inputs and outputs from the witness vector.
            // Snarkjs prove command will generate a public.json file that contains the public inputs and outputs.
            // So if the sizes are not correct, the public.json will not be generated correctly, which might potential contains the private inputs.
            public_outputs: self.r1cs_backend.public_outputs.len(),
            public_inputs: self.r1cs_backend.public_inputs.len(),
            // There seems no use of this field in the snarkjs lib. It might be just a reference.
            private_inputs: self.r1cs_backend.private_input_number(),
            // Add one to take into account the first var that is only added during the witness formation for snarkjs.
            total_wires: self.r1cs_backend.witness_vector.len(),
            // This is for circom lang debugging, so we don't need it.
            number_of_labels: 0,
            number_of_constraints: restructure_constraints.len(),
        };
        header_section
            .write_section(header_data)
            .map_err(|_| Error::R1CSWriterIo)?;

        let r1cs = header_section.end_section().unwrap();
        R1CSWriter::finish_writing(r1cs).map_err(|_| Error::R1CSWriterIo)
    }

    /// Generate the wtns file in snarkjs format.
    pub fn gen_wtns_file(&self, file: &str, witness: GeneratedWitness<F>) -> Result<(), Error> {
        let restructured_witness = self.restructure_witness(witness);

        let mut witness_writer = WitnessWriter::new(file).unwrap();

        Ok(witness_writer.write(restructured_witness, &self.backend_prime())?)
    }

    fn backend_prime(&self) -> BigInt {
        BigInt::from_bytes_le(
            num_bigint_dig::Sign::Plus,
            &self.r1cs_backend.prime().to_bytes_le(),
        )
    }

    fn convert_to_bigint(value: &F) -> BigInt {
        BigInt::from_bytes_le(
            num_bigint_dig::Sign::Plus,
            &ark_ff::BigInteger::to_bytes_le(&value.into_repr()),
        )
    }
}

/// Witness writer for generating the wtns file in snarkjs format.
/// The circom rust lib seems not to have the API to generate the wtns file, so we create our own based on the snarkjs lib.
/// The implementation follows: https://github.com/iden3/snarkjs/blob/577b3f358016a486402050d3b7242876082c085f/src/wtns_utils.js#L25
/// It uses the same binary format as the r1cs file relies on.
/// Although it is the same binary file protocol, but the witness file has simpler data points than the r1cs file.
struct WitnessWriter {
    inner: BufWriter<File>,
    writing_section: Option<WritingSection>,
    section_size_position: u64,
}

/// A struct to represent a section in the snarkjs binary format.
struct WritingSection;

impl WitnessWriter {
    // Initialize a FileWriter
    fn new(path: &str) -> Result<WitnessWriter, Error> {
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
        assert!(
            file_type_bytes.len() == 4,
            "File type must be 4 characters long"
        );
        writer.write_all(file_type_bytes);

        // Write the version as a 32-bit unsigned integer in little endian
        writer.write_all(&version.to_le_bytes())?;

        // Write the number of sections as a 32-bit unsigned integer in little endian
        writer.write_all(&n_sections.to_le_bytes())?;

        let current_position = writer.stream_position().unwrap();

        Ok(WitnessWriter {
            inner: writer,
            writing_section: None,
            section_size_position: current_position,
        })
    }

    /// Start a new section for writing.
    fn start_write_section(&mut self, id_section: u32) -> Result<(), io::Error> {
        // Write the section ID as ULE32
        self.inner.write_all(&id_section.to_le_bytes())?;
        // Get the current position
        self.section_size_position = self.inner.stream_position().unwrap();
        // Temporarily write 0 as ULE64 for the section size
        self.inner.write_all(&0u64.to_le_bytes())?;
        self.writing_section = Some(WritingSection);

        Ok(())
    }

    /// End the current section
    fn end_write_section(&mut self) -> Result<(), io::Error> {
        let current_pos = self.inner.stream_position().unwrap();
        // Calculate the size of the section
        let section_size = current_pos - self.section_size_position - 8;

        // Move back to where the size needs to be written
        self.inner
            .seek(SeekFrom::Start(self.section_size_position))?;
        // Write the actual section size
        self.inner.write_all(&section_size.to_le_bytes())?;
        // Return to the end of the section
        self.inner.seek(SeekFrom::Start(current_pos))?;
        // Flush the buffer to ensure all data is written to the file
        self.inner.flush()?;

        self.writing_section = None;

        Ok(())
    }

    /// Write the witness to the file
    /// It stores the two sections:
    /// - Header section: describes the field size, prime field, and the number of witnesses.
    /// - Witness section: contains the witness values.
    fn write(&mut self, witness: Vec<BigInt>, prime: &BigInt) -> Result<(), io::Error> {
        // Start writing the first section
        self.start_write_section(1)?;
        // Write field size in number of bytes
        let field_n_bytes = field_size(prime);
        self.inner
            .write_all(&(field_n_bytes as u32).to_le_bytes())?;
        // Write the prime field in bytes
        self.write_big_int(prime.clone(), field_n_bytes)?;
        // Write the number of witnesses
        self.inner
            .write_all(&(witness.len() as u32).to_le_bytes())?;

        self.end_write_section()?;

        // Start writing the second section
        self.start_write_section(2)?;

        for value in witness {
            self.write_big_int(value.clone(), field_n_bytes)?;
        }
        self.end_write_section()
    }

    /// Write a `BigInt` to the file
    fn write_big_int(&mut self, value: BigInt, size: usize) -> Result<(), io::Error> {
        let bytes = value.to_bytes_le().1;

        let mut buffer = vec![0u8; size];
        buffer[..bytes.len()].copy_from_slice(&bytes);
        self.inner.write_all(&buffer)
    }
}
