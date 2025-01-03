use super::test_stdlib;
use crate::error::Result;
use rstest::rstest;

#[rstest]
#[case::empty_insert(
    "111",
   "222",
    vec![],
    vec![],
    "0",
    "0",
    "0",
    true,
    true,
    false,
    "10729541595941744696255200734832925648647334864637393545770039840405438557214"
)]
#[case::insert(
    "9700",
    "8800",
    vec![
        "11516567903461282088126784254894078034845453066905529710360444678685967109731",
        "20659815156440161169257728848234717083009297735150715641331129637520746075208",
        "9913888994783849052153109228696667088493732024315564088242255883696537997181",
        "5846636767743912144636001040109151354569603553960606470003432371719273838746",
        "11187195831226248797031600213284009205392316257240308867007036461570604014316",
        "14862507023886888124421673098712328314603512225968198646585391849514658546416"
    ],
    vec![0 , 1, 2, 3, 4, 5],
    "17581302140303159455912973258196037026284300302708949996108423583963947226858",
    "3492",
    "3168",
    false,
    true,
    false,
    "17713115569734927279694966589149598343072771273196461675493145305311861382883"
)]
#[case::delete(
    "111",
    "222",
    vec!["9220985749551237028296517339018427057953245762011653459076210336571800515245"],
    vec![1],
    "21805692344774694518236557976212317500166230062431619316104692181032186605312",
    "555",
    "666",
    false,
    true,
    true,
    "3039938863220546817637150518308754073715763397170404924604005494776416658714"
)]
#[case::update(
    "555",
    "777",
    vec!["9220985749551237028296517339018427057953245762011653459076210336571800515245"],
    vec![1],
    "3039938863220546817637150518308754073715763397170404924604005494776416658714",
    "555",
    "666",
    false,
    false, // update operation
    true,
    "18811865073273086230239721237564240209328819936273238864031238045766843861603"
)]
fn test_smt_cr(
    #[case] key: &str,
    #[case] val: &str,
    #[case] non_zero_sibling: Vec<&str>,
    #[case] non_zero_sibling_index: Vec<usize>,
    #[case] old_root: &str,
    #[case] old_key: &str,
    #[case] old_val: &str,
    #[case] is_old_zero: bool,
    #[case] operation_1: bool,
    #[case] operation_2: bool,
    #[case] expected_new_root: &str,
) -> Result<()> {
    let mut siblings = vec!["0"; 254];

    for (&index, sibling) in non_zero_sibling_index.iter().zip(non_zero_sibling) {
        siblings[index] = sibling;
    }

    let public_inputs = format!(
        r#"{{"siblings": {:?}}}"#,
        siblings
            .iter()
            .map(|ele| ele.to_string())
            .collect::<Vec<String>>()
    );

    let mut values = vec!["0"; 8]; // 0 , 1 -> key ,val
    values[0] = old_root;
    values[1] = key;
    values[2] = val;
    values[3] = old_key;
    values[4] = old_val;
    values[5] = if is_old_zero { "0" } else { "1" };
    values[6] = if operation_1 { "0" } else { "1" };
    values[7] = if operation_2 { "0" } else { "1" };

    let private_inputs = format!(
        r#"{{"values": {:?}}}"#,
        values
            .iter()
            .map(|ele| ele.to_string())
            .collect::<Vec<String>>()
    );

    test_stdlib(
        "smt/smt_main.no",
        None,
        &public_inputs,
        &private_inputs,
        vec![expected_new_root],
    )?;
    Ok(())
}

#[rstest]
#[case::inclusion(
    "333",
    "444",
    vec![
        "15403437905133579310679669358298285751036324375519574557984500284974195012647",
        "16164523410687895121172017182256869209088533188202760284238496207325271948775"
    ],
    vec![0,1],
    "12941802777540120349830076641367475813359582080712967896858975038182858131027",
    "0",
    "0",
    false,
    true,
)]
#[case::exclusion(
    "1000",
    "0",
    vec![
        "5004112844904397918413167045606564570413835725979211272408079893204730422053",
        "20870930364208425904173849538077157932823107157308028777399852634164408184090",
        "18817399965850323578786675877025159015083291330173277928593283742904067184537",
        "11475507857885337462985742557005542752995566264656500462775773988511382189430",
        "20591657041708931641763347242286558192823550076918037187689654551559790721676",
        "8391178249010813208860647414946215155510772994793073739371291818862143236795"
    ],
    vec![0,1,2,3,4,5],
    "12941802777540120349830076641367475813359582080712967896858975038182858131027",
    "1960",
    "1760",
    false,
    false,
)]
fn test_smt_ie(
    #[case] key: &str,
    #[case] val: &str,
    #[case] non_zero_sibling: Vec<&str>,
    #[case] non_zero_sibling_index: Vec<usize>,
    #[case] root: &str,
    #[case] not_found_key: &str,
    #[case] not_found_val: &str,
    #[case] is_old_zero: bool,
    #[case] inclusion_proof: bool,
) -> Result<()> {
    let mut siblings = vec!["0"; 254];

    for (&index, sibling) in non_zero_sibling_index.iter().zip(non_zero_sibling) {
        siblings[index] = sibling;
    }

    let public_inputs = format!(
        r#"{{"siblings": {:?}}}"#,
        siblings
            .iter()
            .map(|ele| ele.to_string())
            .collect::<Vec<String>>()
    );

    let mut values = vec!["0"; 7]; // 0 , 1 -> key ,val
    values[0] = root;
    values[1] = not_found_key;
    values[2] = not_found_val;
    values[3] = key;
    values[4] = val;
    values[5] = if is_old_zero { "0" } else { "1" };
    values[6] = if inclusion_proof { "0" } else { "1" };

    let private_inputs = format!(
        r#"{{"values": {:?}}}"#,
        values
            .iter()
            .map(|ele| ele.to_string())
            .collect::<Vec<String>>()
    );

    test_stdlib(
        "smt/smt_verify.no",
        None,
        &public_inputs,
        &private_inputs,
        vec![],
    )?;
    Ok(())
}
